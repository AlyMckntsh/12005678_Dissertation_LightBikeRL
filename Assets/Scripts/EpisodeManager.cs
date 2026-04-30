using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Runs the round step by step. It moves both bikes together, works out who crashed,
/// assigns the round result, and resets the arena for the next round.
/// This script is the shared clock for the arena, so both bikes always resolve one grid move together.
/// </summary>
[DefaultExecutionOrder(100)]
[DisallowMultipleComponent]
public class EpisodeManager : MonoBehaviour
{
    /// <summary>
    /// Stores the result of the finished round.
    /// </summary>
    internal enum RoundOutcome
    {
        None = 0,
        LearnerWin = 1,
        OpponentWin = 2,
        Draw = 3
    }

    /// <summary>
    /// Stores what one bike wants to do on the next shared step.
    /// </summary>
    private struct StepIntent
    {
        public PlayerBikeController Bike;
        public Vector2Int CurrentCell;
        public Vector2Int NextCell;
        public PlayerBikeController.CollisionReason CollisionReason;
    }

    // Keep one authoritative simulation step per frame update so timing stays predictable.
    private const int MaxSimulationStepsPerFrame = 1;

    [SerializeField] private ArenaContext arenaContext;

    // _arenaBikes is the live arena snapshot. _stepIntents is the per-step move plan built from it.
    private readonly List<PlayerBikeController> _arenaBikes = new List<PlayerBikeController>(2);
    private readonly List<StepIntent> _stepIntents = new List<StepIntent>(2);
    private bool _isEndingEpisode;
    private bool _isRoundPlaying = true;
    private float _stepTimer;

    private void Awake()
    {
        if (arenaContext == null)
        {
            Debug.LogError("[EpisodeManager] ArenaContext reference is required.", this);
            enabled = false;
        }
    }

    private void Update()
    {
        if (!enabled || _isEndingEpisode || !_isRoundPlaying)
        {
            return;
        }

        // Accumulate real time until it is time to advance one shared grid step.
        if (!GetStepDuration(out float stepDuration))
        {
            return;
        }

        _stepTimer += Time.deltaTime;

        int stepsTaken = 0;
        while (_stepTimer >= stepDuration && stepsTaken < MaxSimulationStepsPerFrame)
        {
            // Consume at most one logical grid step this frame, even if rendering slows down.
            _stepTimer -= stepDuration;
            SimulateSynchronizedStep();
            stepsTaken++;
        }

        if (_stepTimer > stepDuration)
        {
            // Clamp any leftover backlog so timing errors do not snowball across later frames.
            _stepTimer = stepDuration;
        }
    }

    internal void HandleBikeDeath(PlayerBikeController deadBike)
    {
        if (!enabled || _isEndingEpisode || deadBike == null)
        {
            return;
        }

        if (deadBike.ArenaContext != arenaContext)
        {
            Debug.LogWarning("[EpisodeManager] Ignoring bike death from a different ArenaContext.", deadBike);
            return;
        }

        // The death only starts end-of-round handling. The actual win/loss result is resolved after both bikes
        // have had a chance to finish the shared step and register their crash state.
        // Only start the end-of-round sequence once, even if both bikes crash together.
        _isEndingEpisode = true;
        _isRoundPlaying = false;
        _stepTimer = 0f;
        StartCoroutine(FinishRoundAndReset());
    }

    private IEnumerator FinishRoundAndReset()
    {
        // Wait one frame so both bikes have finished any crash side effects.
        yield return null;

        try
        {
            arenaContext.CopyActiveBikesTo(_arenaBikes);

            // Resolve the winner before the reset so terminal rewards reflect the final board state.
            RoundOutcome roundOutcome = ResolveRoundOutcome();

            for (int i = 0; i < _arenaBikes.Count; i++)
            {
                CompleteBikeEpisode(_arenaBikes[i], roundOutcome);
            }

            // Reset the whole arena from one place so both bikes start the next round together.
            arenaContext.TriggerRoundReset();
            _isRoundPlaying = true;
        }
        finally
        {
            _isEndingEpisode = false;
        }
    }

    private RoundOutcome ResolveRoundOutcome()
    {
        PlayerBikeController learnerBike = null;
        PlayerBikeController opponentBike = null;

        // Find the learner/opponent pair currently registered in this arena.
        for (int i = 0; i < _arenaBikes.Count; i++)
        {
            PlayerBikeController bike = _arenaBikes[i];
            if (bike == null)
            {
                continue;
            }

            if (bike.IsOpponentRole)
            {
                opponentBike = bike;
            }
            else
            {
                learnerBike = bike;
            }
        }

        bool learnerAlive = learnerBike != null && !learnerBike.IsDead;
        bool opponentAlive = opponentBike != null && !opponentBike.IsDead;

        if (learnerAlive && opponentAlive)
        {
            // Defensive fallback: normally HandleBikeDeath means at least one bike is already dead here.
            return RoundOutcome.None;
        }

        if (!learnerAlive && !opponentAlive)
        {
            return RoundOutcome.Draw;
        }

        if (learnerAlive)
        {
            return RoundOutcome.LearnerWin;
        }

        if (opponentAlive)
        {
            return RoundOutcome.OpponentWin;
        }

        return RoundOutcome.Draw;
    }

    private void CompleteBikeEpisode(PlayerBikeController bike, RoundOutcome roundOutcome)
    {
        if (bike == null)
        {
            return;
        }

        // First work out whether this bike won, lost, or drew.
        LightBikeAgent.EpisodeOutcome episodeOutcome = ResolveEpisodeOutcome(bike, roundOutcome);

        LightBikeAgent agent = bike.GetComponent<LightBikeAgent>();
        if (agent == null || !agent.isActiveAndEnabled)
        {
            return;
        }

        RewardManager rewardManager = bike.GetComponent<RewardManager>();
        if (rewardManager == null)
        {
            Debug.LogError($"[EpisodeManager] Bike {bike.name} has an active LightBikeAgent but no RewardManager.", bike);
            return;
        }

        // RewardManager owns the terminal reward values for active agent bikes.
        float terminalReward = rewardManager.EvaluateTerminalReward(episodeOutcome);
        agent.CompleteEpisodeFromManager(episodeOutcome, terminalReward);
    }

    private static LightBikeAgent.EpisodeOutcome ResolveEpisodeOutcome(
        PlayerBikeController bike,
        RoundOutcome roundOutcome)
    {
        // Convert the arena-level result into the learner/opponent view for this specific bike.
        if (roundOutcome == RoundOutcome.Draw || roundOutcome == RoundOutcome.None)
        {
            return LightBikeAgent.EpisodeOutcome.Draw;
        }

        bool didWin = false;
        if (roundOutcome == RoundOutcome.LearnerWin && !bike.IsOpponentRole)
        {
            didWin = true;
        }

        if (roundOutcome == RoundOutcome.OpponentWin && bike.IsOpponentRole)
        {
            didWin = true;
        }

        if (didWin)
        {
            return LightBikeAgent.EpisodeOutcome.Win;
        }

        return LightBikeAgent.EpisodeOutcome.Loss;
    }

    private bool GetStepDuration(out float stepDuration)
    {
        stepDuration = 0f;
        arenaContext.CopyActiveBikesTo(_arenaBikes);

        float longestStepDuration = 0f;
        for (int i = 0; i < _arenaBikes.Count; i++)
        {
            PlayerBikeController bike = _arenaBikes[i];
            if (bike != null && !bike.IsDead)
            {
                float bikeStepDuration = Mathf.Max(0.001f, bike.StepDuration);
                longestStepDuration = Mathf.Max(longestStepDuration, bikeStepDuration);
            }
        }

        if (longestStepDuration <= 0f)
        {
            return false;
        }

        // Drive the arena on the slowest alive bike so the shared step stays safe and synchronized.
        stepDuration = longestStepDuration;
        return true;
    }

    private void SimulateSynchronizedStep()
    {
        arenaContext.CopyActiveBikesTo(_arenaBikes);
        _stepIntents.Clear();

        // Ask every living bike what cell it wants to move into next.
        for (int i = 0; i < _arenaBikes.Count; i++)
        {
            PlayerBikeController bike = _arenaBikes[i];
            if (bike == null || bike.IsDead)
            {
                continue;
            }

            Vector2Int nextCell = bike.PrepareStepIntent();
            PlayerBikeController.CollisionReason collisionReason = bike.ResolveBaseCollisionReason(nextCell);

            // Store both the current and next cells so later conflict checks can detect same-cell contests and head swaps.
            _stepIntents.Add(new StepIntent
            {
                Bike = bike,
                CurrentCell = bike.CurrentCell,
                NextCell = nextCell,
                CollisionReason = collisionReason
            });
        }

        if (_stepIntents.Count <= 0)
        {
            return;
        }

        // After the basic wall and trail checks, compare the bikes against each other.
        ResolveMoveConflicts();

        // Finally apply crashes and moves in a fixed order.
        CommitStepResults();
    }

    private void ResolveMoveConflicts()
    {
        // Compare every pair of move intents for collisions that only appear once both bikes are considered together.
        for (int i = 0; i < _stepIntents.Count; i++)
        {
            for (int j = i + 1; j < _stepIntents.Count; j++)
            {
                ResolveSameCellContest(i, j);
                ResolveHeadSwap(i, j);
            }
        }
    }

    private void CommitStepResults()
    {
        // First crash any bike that lost the move.
        for (int i = 0; i < _stepIntents.Count; i++)
        {
            StepIntent intent = _stepIntents[i];
            if (intent.CollisionReason != PlayerBikeController.CollisionReason.None)
            {
                intent.Bike.CommitResolvedCrash(intent.CollisionReason);
            }
        }

        // Then move the bikes that survived.
        // This second pass matters because one bike crashing should not let the other one "claim" a shared conflict cell early.
        for (int i = 0; i < _stepIntents.Count; i++)
        {
            StepIntent intent = _stepIntents[i];
            if (intent.CollisionReason == PlayerBikeController.CollisionReason.None)
            {
                intent.Bike.CommitResolvedMove(intent.NextCell);
            }
        }

        // Tell the surviving bikes that the shared step is finished.
        for (int i = 0; i < _stepIntents.Count; i++)
        {
            StepIntent intent = _stepIntents[i];
            if (intent.CollisionReason == PlayerBikeController.CollisionReason.None)
            {
                intent.Bike.NotifyStepAdvanced();
            }
        }
    }

    private void ResolveSameCellContest(int firstIndex, int secondIndex)
    {
        if (_stepIntents[firstIndex].NextCell != _stepIntents[secondIndex].NextCell)
        {
            return;
        }

        // If both bikes want the same cell on the same step, both of them lose that move.
        StepIntent first = _stepIntents[firstIndex];
        if (first.CollisionReason == PlayerBikeController.CollisionReason.None)
        {
            first.CollisionReason = PlayerBikeController.CollisionReason.SameCellContest;
            _stepIntents[firstIndex] = first;
        }

        StepIntent second = _stepIntents[secondIndex];
        if (second.CollisionReason == PlayerBikeController.CollisionReason.None)
        {
            second.CollisionReason = PlayerBikeController.CollisionReason.SameCellContest;
            _stepIntents[secondIndex] = second;
        }
    }

    private void ResolveHeadSwap(int firstIndex, int secondIndex)
    {
        StepIntent first = _stepIntents[firstIndex];
        StepIntent second = _stepIntents[secondIndex];

        bool isHeadSwap = first.NextCell == second.CurrentCell &&
                          second.NextCell == first.CurrentCell;
        if (!isHeadSwap)
        {
            return;
        }

        // A head swap means the bikes crossed through each other on the same step,
        // so both should be marked as crashed.
        if (first.CollisionReason == PlayerBikeController.CollisionReason.None ||
            first.CollisionReason == PlayerBikeController.CollisionReason.OpponentHead)
        {
            first.CollisionReason = PlayerBikeController.CollisionReason.HeadSwap;
            _stepIntents[firstIndex] = first;
        }

        if (second.CollisionReason == PlayerBikeController.CollisionReason.None ||
            second.CollisionReason == PlayerBikeController.CollisionReason.OpponentHead)
        {
            second.CollisionReason = PlayerBikeController.CollisionReason.HeadSwap;
            _stepIntents[secondIndex] = second;
        }
    }

}
