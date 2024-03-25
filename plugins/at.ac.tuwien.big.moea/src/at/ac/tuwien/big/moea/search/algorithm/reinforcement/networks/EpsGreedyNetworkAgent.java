package at.ac.tuwien.big.moea.search.algorithm.reinforcement.networks;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.UnitParameter;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.moeaframework.core.Problem;
import org.moeaframework.core.Solution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;

public class EpsGreedyNetworkAgent<S extends Solution> extends AbstractNetworkAgent<S> {

   public class Replay {

      boolean done;
      int action;
      INDArray[] oldObs;
      INDArray[] nextObs;
      double reward;

      // response recieved, obs we saw and action we took
      public Replay(final boolean done, final double reward, final INDArray[] nextObs, final INDArray[] oldObs,
            final int action) {
         this.done = done;
         this.action = action;
         this.oldObs = oldObs;
         this.nextObs = nextObs;
         this.reward = reward;
      }

      public int action() {
         return action;
      }

      public boolean done() {
         return done;
      }

      public INDArray[] nextObs() {
         return nextObs;
      }

      public INDArray[] prevObs() {
         return oldObs;
      }

      public double reward() {
         return reward;
      }

   }

   private final INetwork nn;
   private final ComputationGraph targetNN;
   private final double epsDecay;
   private double epsilon;
   private final int numEpisodes;
   private final Random rand;
   private final int batchSize;
   private final int bufferSize;
   private final int updateTarget;
   private final double epsilonLB;
   private final double epsilonUB;
   private final List<Replay> replays = new ArrayList<>();

   public EpsGreedyNetworkAgent(final INetwork nn, final ComputationGraph targetNN, final double epsilon,
         final int numEpisodes, final int bufferSize, final int batchSize, final int updateTarget,
         final Problem problem, final ISOEnvironment<S> environment, final String scoreSavePath,
         final int epochsPerModelSave, final int terminateAfterEpisodes, final boolean verbose) {
      super(problem, environment, scoreSavePath, terminateAfterEpisodes, epochsPerModelSave, verbose);
      this.nn = nn;
      this.targetNN = targetNN;
      this.numEpisodes = numEpisodes;
      this.epsilon = epsilon;
      this.epsDecay = Math.pow(10, 4);
      this.epsilonLB = 0.05;
      this.epsilonUB = 0.8;
      this.rand = new Random(12);
      this.batchSize = batchSize;
      this.bufferSize = bufferSize;
      this.updateTarget = updateTarget;

   }

   @Override
   protected void iterate() {
      if(this.startTime == 0) {
         this.startTime = System.currentTimeMillis();
      }

      this.trainEpoch(100);

      // if(saveInterval > 0 && nrOfEpochs % saveInterval == 0) {
      // nn.saveModel(nrOfEpochs);
      // }
      //
      // if(terminateAfterEpisodes > 0 && terminateAfterEpisodes <= nrOfEpochs) {
      // this.terminate();
      // System.out.println("Terminated after " + nrOfEpochs + " epochs");
      // nn.saveFinalModel();
      // }
      //
      // if(this.scoreSavePath != null && saveInterval > 0 && nrOfEpochs % saveInterval == 0) {
      // System.out.println("Saving rewards at epoch " + nrOfEpochs + " after "
      // + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
      // saveRewards(framesList, rewardEarned, meanRewardEarned, timePassedList, nrOfEpochs);
      // }

   }

   public void performFit(final List<Replay> replays, final int actionSpace) {

      final INDArray[] features = new INDArray[replays.get(0).oldObs.length];
      final INDArray labels = Nd4j.zeros(batchSize, actionSpace);

      for(int i = 0; i < features.length; i++) {
         features[i] = Nd4j.zeros(batchSize, replays.get(0).oldObs[0].length());
      }

      final List<Integer> indices = new ArrayList<>();
      for(int i = 0; i < batchSize; i++) {

         final int index = rand.nextInt(replays.size());
         indices.add(index);

         final INDArray predQ = nn.outputSingle(replays.get(index).prevObs()); // get our prediction
         // final INDArray futNetQ = nn.outputSingle(replays.get(index).nextObs());

         // use the final reward if game over or the max reward from the target net
         double label;
         int action;
         final int multi = replays.get(index).done() ? 0 : 1;

         action = (int) (long) Nd4j.getExecutioner().execAndReturn(new IMax(predQ)).getFinalResult();

         final INDArray futureQ = targetNN.outputSingle(replays.get(index).nextObs());

         label = replays.get(index).reward() + nn.getGamma() * ((double) futureQ.maxNumber() * multi); // get value

         predQ.putScalar(0, action, label); // place the correct reward at correct index leave all else

         for(int j = 0; j < features.length; j++) {
            features[j].putRow(i, replays.get(index).oldObs[j]);
         }

         labels.putRow(i, Nd4j.toFlattened(predQ));

      }
      final INDArray[] tempArr = { labels };

      nn.getComputationGraph().fit(features, tempArr);

   }

   public void trainEpoch(final int update_after) {

      // double oldReward = 2.82842;
      INDArray dist;

      final S initialSolution = environment.reset();
      if(this.verbose) {
         this.encoder.printModel(initialSolution);
      }
      INDArray[] oldObs = this.encoder.encodeSolution(initialSolution);
      S oldSolution = environment.getInitialSolution();

      // final ArrayList<INDArray[]> stateLs = new ArrayList<>();
      // final ArrayList<Integer> actionLs = new ArrayList<>();
      // final ArrayList<INDArray> rewardLs = new ArrayList<>();
      final boolean maxReached = false;

      printIfVerboseMode("Starting training epoch " + nrOfEpochs++ + "..");
      int episodeNum = 0;

      final Deque<Double> ret25 = new ArrayDeque<>(25);

      while(episodeNum < this.numEpisodes) {

         double ret = 0;
         boolean done = false;
         while(!done) {

            epsilon = Math.max(epsilonLB, epsilonUB - framecount / epsDecay);

            dist = nn.outputSingle(oldObs);

            if(this.verbose || epsilon < .3) {
               this.encoder.printModel(oldSolution);
               System.out.println(this.encoder.getDistrActionProbabilities(dist.dup(), oldSolution, false));
            }

            int action = -1;
            final List<Integer> legalActions = this.encoder.getLegalActions(oldSolution);

            if(rand.nextDouble() < epsilon) {
               action = legalActions.get(rand.nextInt(legalActions.size()));

            } else {
               action = (int) (long) Nd4j.getExecutioner()
                     .execAndReturn(new IMax(this.encoder.getQValueDistributionForLegalActions(dist.dup(),
                           Double.NEGATIVE_INFINITY, legalActions, this.encoder.getActionSpace(oldSolution))))
                     .getFinalResult();
            }

            // printIfVerboseMode(this.encoder.getDistrActionProbabilities(dist.dup(), oldSolution, true));

            final SOEnvResponse<S> response = environment.step(oldSolution, action);

            // final int action = response.getAppliedActionId();

            final S newSolution = response.getState();

            this.addSolutionIfImprovement(newSolution);

            ret += response.getReward();

            final double[] curReward = new double[1];
            curReward[0] = response.getReward();

            final INDArray[] nextObs = this.encoder.encodeSolution(newSolution);

            if(replays.size() == this.bufferSize) {
               replays.remove(rand.nextInt(this.bufferSize));
            }

            replays.add(new Replay(response.getDoneStatus() != null, response.getReward(), nextObs, oldObs, action));

            oldSolution = newSolution;
            oldObs = nextObs;
            done = response.getDoneStatus() != null;

            if(replays.size() > batchSize && replays.size() > update_after) {
               this.performFit(replays, this.encoder.getActionSpace(newSolution));
            }

            if(framecount % updateTarget == 0) {
               this.updateTarget(targetNN, nn.getComputationGraph()); // update other network
               System.out.println("Target Update @ " + framecount);
            }

            framecount++;

            // stateLs.add(oldObs);
            // actionLs.add(action);

            final UnitParameter up = this.encoder.getFixedUnitApplicationStrategy(oldSolution, action)
                  .getDistributionSampleRule();
            printIfVerboseMode("\nAction => "
                  + String.format("%d (%s -> %s)", action, up.getUnitName(), up.getParameterValues().toString()));
            printIfVerboseMode("Reward =>=> " + Arrays.toString(curReward));

            if(done) {
               oldSolution = environment.reset();
               if(this.verbose) {
                  this.encoder.printModel(oldSolution);
               }
               oldObs = this.encoder.encodeSolution(environment.getInitialSolution());

               rewardEarned.add(ret);
               framesList.add((double) framecount);
               timePassedList.add((double) (System.currentTimeMillis() - startTime));

               System.out.println("Reward: " + ret + "\tEps: " + epsilon);
            }
         }

         if(saveInterval > 0 && episodeNum % saveInterval == 0) {
            nn.saveModel(episodeNum);
         }

         ret25.add(ret);

         if(episodeNum % 25 == 0) {
            System.out.println("Episode " + episodeNum + "\tAverage Reward: "
                  + ret25.stream().mapToDouble(x -> x).average().getAsDouble());
         }
         episodeNum++;
      }

      // nn.fit(stateLs, actionLs, rewardLs);
      //
      // rewardEarned.add(cumReward);
      // framesList.add((double) framecount);

      // printIfVerboseMode("Avg. reward: " + cumReward / epochSteps);

   }

   public void updateTarget(final ComputationGraph target, final ComputationGraph net) {

      // extract the network params to gen so that it has its new learned strategy
      for(int i = 0; i < net.getLayers().length; i++) {
         target.getLayer(i).setParams(net.getLayer(i).params());
      }

   }

   // public void updateTarget(final ComputationGraph target, final ComputationGraph net, final double tau) {
   //
   // // extract the network params to gen so that it has its new learned strategy
   // for(int i = 0; i < net.getLayers().length; i++) {
   // final INDArray updParams = net.getLayer(i).params().mul(tau).add(target.getLayer(i).params().mul(1 - tau));
   // System.out.println("Updated params: " + updParams.toStringFull());
   // target.getLayer(i).setParams(updParams);
   // }
   //
   // }
}
