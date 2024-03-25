package at.ac.tuwien.big.moea.search.algorithm.reinforcement.algorithms;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.AbstractMOTabularRLAgent;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.IMOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.MOEnvResponse;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.LocalSearchStrategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.moeaframework.core.Problem;
import org.moeaframework.core.Solution;

public class GoExplore<S extends Solution> extends AbstractMOTabularRLAgent<S> {

   // private final double gamma; // Eagerness - 0 looks in the near future, 1 looks in the distant future
   // private final double eps;
   private final int exploreSteps;
   private final int maxIterations;
   private final Map<Integer, S> q;
   private final LocalSearchStrategy localSearchStrategy;

   public GoExplore(final LocalSearchStrategy localSearchStrategy, final int exploreSteps, final Problem problem,
         final IMOEnvironment<S> environment, final String savePath, final int recordInterval,
         final int terminateAfterEpisodes, final String qTableIn, final String qTableOut, final boolean verbose) {
      super(problem, environment, savePath, recordInterval, terminateAfterEpisodes, qTableIn, qTableOut, verbose);

      this.exploreSteps = exploreSteps;
      this.localSearchStrategy = localSearchStrategy;
      this.maxIterations = iterations;

      this.q = new HashMap<>();

   }

   @Override
   public List<IApplicationState> epsGreedyDecision() {
      return null;
   }

   /*
    * # Take action, get state, reward and if end state
    * next_state, reward, done = env.step(action)
    * # Update benefit of taken action
    * Q[state][action] = Q[state][action] + alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])
    */
   @Override
   protected void iterate() {

      final MOEnvResponse<S> response = (MOEnvResponse<S>) environment.step(localSearchStrategy, currentSolution,
            exploreSteps);

      final S nextState = response.getState();
      final DoneStatus doneStatus = response.getDoneStatus();

      if(!q.containsKey(epochSteps + 1) && !this.isDominatedByAnySolutionInParetoFront(nextState)) {
         q.put(epochSteps + 1, nextState);
         this.population.add(nextState);
      } else {
         if(!q.containsKey(epochSteps + 1)
               || this.population.getComparator().compare(nextState, q.get(epochSteps + 1)) <= 0) {
            q.put(epochSteps + 1, nextState);
            this.population.add(nextState);
         }
      }

      iterations++;

      if(doneStatus == DoneStatus.MAX_LENGTH_REACHED || doneStatus == DoneStatus.FINAL_STATE_REACHED) {
         if(this.saveInterval > 0) {

            for(int i = 0; i < rewardEarnedLists.size(); i++) {
               rewardEarnedLists.get(i).add(cumRewardList.get(i));
               meanRewardEarnedLists.get(i).add(cumRewardList.get(i) / epochSteps);
            }
            framesList.add((double) iterations);
            timePassedList.add((double) (System.currentTimeMillis() - startTime));

            if(scoreSavePath != null && nrOfEpochs > 0 && nrOfEpochs % this.saveInterval == 0) {

               saveRewards(scoreSavePath, framesList, this.environment.getFunctionNames(), rewardEarnedLists,
                     timePassedList, meanRewardEarnedLists);
            }
         }

         nrOfEpochs++;
         cumRewardList = new ArrayList<>(Collections.nCopies(this.environment.getFitnessComparators().size(), 0.0));
         epochSteps = 0;
         currentSolution = environment.reset();
      } else {
         epochSteps++;
         currentSolution = q.get(this.epochSteps);
      }
   }

}
