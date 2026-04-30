using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Connects ML-Agents to the light bike game. It collects observations, receives actions,
/// applies rewards, and reports how each round finished.
/// The flow follows the usual Agent lifecycle: initialise once, reset on episode start,
/// observe, act, then finish the episode after EpisodeManager resolves the round.
/// </summary>
[DisallowMultipleComponent]
[RequireComponent(typeof(PlayerBikeController))]
[RequireComponent(typeof(BehaviorParameters))]
[RequireComponent(typeof(RewardManager))]
[RequireComponent(typeof(DecisionRequester))]
public class LightBikeAgent : Agent
{
    /// <summary>
    /// Stores the tie-break scores for one heuristic move candidate.
    /// </summary>
    private struct HeuristicDirectionEvaluation
    {
        internal int Action;
        internal int ImmediateSafeContinuationCount;
        internal int SecondStepContinuationQuality;
        internal int CenterPreference;
        internal int ForwardPreference;
        internal int TieBreakPriority;
    }

    /// <summary>
    /// Stores the final result of the episode for this bike.
    /// </summary>
    internal enum EpisodeOutcome
    {
        Loss = 0,
        Win = 1,
        Draw = 2
    }

    /// <summary>
    /// Small result object used when one round has finished.
    /// </summary>
    internal struct EpisodeSummary
    {
        public EpisodeSummary(EpisodeOutcome outcome, int episodeLength, float totalReward)
        {
            Outcome = outcome;
            EpisodeLength = episodeLength;
            TotalReward = totalReward;
        }

        public EpisodeOutcome Outcome
        {
            get;
        }

        public bool DidWin
        {
            get { return Outcome == EpisodeOutcome.Win; }
        }

        public bool DidDraw
        {
            get { return Outcome == EpisodeOutcome.Draw; }
        }

        public int EpisodeLength
        {
            get;
        }

        public float TotalReward
        {
            get;
        }
    }

    // Observation layout: 3 immediate block checks, 3 opponent hints, and 3 one-step look-ahead checks.
    private const int ObservationCount = 9;
    // Three discrete actions are exposed to ML-Agents: turn left, keep going straight, or turn right.
    private const int DiscreteActionCount = 3;
    // Keep the default episode cap aligned with the reward tuning reference used by RewardManager.
    private const int DefaultMaxStep = 1500;

    private PlayerBikeController _bike;
    private BehaviorParameters _behaviorParameters;
    private RewardManager _rewardManager;
    private bool _isSubscribedToBikeSteps;

    // MatchController listens for this so it can record finished learner episodes without polling the agent.
    internal event Action<LightBikeAgent, EpisodeSummary> EpisodeCompleted;

    protected override void Awake()
    {
        base.Awake();
        CacheReferences();
        SubscribeToBikeStepEvents();
    }

    protected override void OnEnable()
    {
        base.OnEnable();
        CacheReferences();
        SubscribeToBikeStepEvents();
    }

    protected override void OnDisable()
    {
        UnsubscribeFromBikeStepEvents();
        base.OnDisable();
    }

    /// <summary>
    /// Finds the required components and prepares the agent for training or inference.
    /// </summary>
    public override void Initialize()
    {
        CacheReferences();
        SubscribeToBikeStepEvents();

        if (!ValidateInferenceModelAssigned())
        {
            return;
        }

        if (MaxStep <= 0)
        {
            // Use the shared default so episode length and reward scaling stay in sync across runs.
            MaxStep = DefaultMaxStep;
        }
    }

    /// <summary>
    /// Resets the bike control state and reward tracking at the start of an episode.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        CacheReferences();
        if (_bike == null)
        {
            return;
        }

        if (!ValidateInferenceModelAssigned())
        {
            return;
        }

        // Start each round with no buffered turn so the reset frame cannot replay an old action.
        _bike.ClearPendingTurn();

        if (_rewardManager != null)
        {
            // The first shaping delta compares against this reset snapshot.
            _rewardManager.BeginEpisode(_bike);
        }
    }

    /// <summary>
    /// Builds the observation vector used by the policy.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        if (_bike == null)
        {
            AddEmptyObservations(sensor);
            return;
        }

        // Work in the bike's local frame so the policy can learn "forward, left, right"
        // instead of memorising absolute world directions.
        GetLocalDirections(out Vector2Int forward, out Vector2Int left, out Vector2Int right);
        Vector2Int head = _bike.CurrentCell;

        AddDirectionBlockObservations(sensor, forward, left, right);
        AddOpponentObservations(sensor, head, forward, right);
        AddLookAheadBlockObservations(sensor, head, forward, left, right);
    }

    /// <summary>
    /// Converts the chosen discrete action into a buffered bike turn.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actions)
    {
        if (_bike == null || _bike.IsDead)
        {
            return;
        }

        int action = 1;
        if (actions.DiscreteActions.Length > 0)
        {
            action = Mathf.Clamp(actions.DiscreteActions[0], 0, DiscreteActionCount - 1);
        }

        // Map ML-Agents action ids 0, 1, and 2 onto left, straight, and right.
        int turn = action - 1;
        _bike.SubmitTurn(turn);
    }

    /// <summary>
    /// Provides a simple fallback move choice for manual testing and heuristic-only modes.
    /// </summary>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
        if (discreteActions.Length <= 0)
        {
            return;
        }

        if (_bike == null || _bike.IsDead)
        {
            discreteActions[0] = 1;
            return;
        }

        GetLocalDirections(out Vector2Int forward, out Vector2Int left, out Vector2Int right);

        bool hasBestEvaluation = false;
        HeuristicDirectionEvaluation bestEvaluation = default;

        // Blocked directions are skipped entirely, so any available safe move beats an immediate collision.
        ConsiderHeuristicDirection(left, 0, 0, 1, ref hasBestEvaluation, ref bestEvaluation);
        ConsiderHeuristicDirection(forward, 1, 1, 0, ref hasBestEvaluation, ref bestEvaluation);
        ConsiderHeuristicDirection(right, 2, 0, 2, ref hasBestEvaluation, ref bestEvaluation);

        discreteActions[0] = hasBestEvaluation ? bestEvaluation.Action : 1;
    }

    internal void CompleteEpisodeFromManager(EpisodeOutcome outcome, float terminalReward)
    {
        CacheReferences();
        if (_bike == null)
        {
            return;
        }

        // Add the final reward first, then report one summary object for the whole round.
        AddReward(terminalReward);
        EpisodeSummary summary = BuildEpisodeSummary(outcome);

        if (EpisodeCompleted != null)
        {
            EpisodeCompleted.Invoke(this, summary);
        }

        EndEpisode();
    }

    // The observation and heuristic code both reason in the bike's local frame.
    private void GetLocalDirections(out Vector2Int forward, out Vector2Int left, out Vector2Int right)
    {
        BikeDirectionUtility.GetRelativeDirections(_bike.CurrentDirection, out forward, out left, out right);
    }

    private void AddDirectionBlockObservations(VectorSensor sensor, Vector2Int forward, Vector2Int left, Vector2Int right)
    {
        // These three values tell the policy whether the immediate forward, left, and right moves are legal.
        sensor.AddObservation(_bike.IsDirectionBlocked(forward) ? 1f : 0f);
        sensor.AddObservation(_bike.IsDirectionBlocked(left) ? 1f : 0f);
        sensor.AddObservation(_bike.IsDirectionBlocked(right) ? 1f : 0f);
    }

    private void AddOpponentObservations(VectorSensor sensor, Vector2Int head, Vector2Int forward, Vector2Int right)
    {
        PlayerBikeController opponent = _bike.FindOpponent();
        if (opponent == null)
        {
            // Keep the observation shape stable even if the opponent is not available yet.
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            return;
        }

        // Convert the opponent position into the bike's local forward/right axes.
        Vector2Int delta = opponent.CurrentCell - head;
        int localForward = delta.x * forward.x + delta.y * forward.y;
        int localRight = delta.x * right.x + delta.y * right.y;

        sensor.AddObservation(CoarseSign(localForward));
        sensor.AddObservation(CoarseSign(localRight));

        float distanceNormalization = GetOpponentDistanceNormalization();
        float distance = (Mathf.Abs(delta.x) + Mathf.Abs(delta.y)) / distanceNormalization;
        sensor.AddObservation(Mathf.Clamp01(distance));
    }

    private void AddLookAheadBlockObservations(
        VectorSensor sensor,
        Vector2Int head,
        Vector2Int forward,
        Vector2Int left,
        Vector2Int right)
    {
        // These three checks tell the policy what the board looks like one step further ahead.
        sensor.AddObservation(_bike.IsCellBlocked(head + (forward * 2)) ? 1f : 0f);
        sensor.AddObservation(_bike.IsCellBlocked(head + forward + left) ? 1f : 0f);
        sensor.AddObservation(_bike.IsCellBlocked(head + forward + right) ? 1f : 0f);
    }

    private void ConsiderHeuristicDirection(
        Vector2Int direction,
        int action,
        int forwardPreference,
        int tieBreakPriority,
        ref bool hasBestEvaluation,
        ref HeuristicDirectionEvaluation bestEvaluation)
    {
        if (_bike.IsDirectionBlocked(direction))
        {
            return;
        }

        // Score one legal move candidate, then keep it if it beats the current best option.
        HeuristicDirectionEvaluation evaluation = EvaluateHeuristicDirection(direction, action, forwardPreference, tieBreakPriority);
        if (!hasBestEvaluation || IsBetterHeuristicEvaluation(evaluation, bestEvaluation))
        {
            bestEvaluation = evaluation;
            hasBestEvaluation = true;
        }
    }

    private HeuristicDirectionEvaluation EvaluateHeuristicDirection(
        Vector2Int direction,
        int action,
        int forwardPreference,
        int tieBreakPriority)
    {
        Vector2Int nextCell = _bike.CurrentCell + direction;

        // Prefer moves that still leave room to continue after the next step.
        EvaluateHeuristicContinuations(nextCell, direction, out int immediateSafeContinuationCount, out int secondStepContinuationQuality);

        return new HeuristicDirectionEvaluation
        {
            Action = action,
            ImmediateSafeContinuationCount = immediateSafeContinuationCount,
            SecondStepContinuationQuality = secondStepContinuationQuality,
            CenterPreference = CalculateCenterPreference(nextCell),
            ForwardPreference = forwardPreference,
            TieBreakPriority = tieBreakPriority
        };
    }

    private void EvaluateHeuristicContinuations(
        Vector2Int originCell,
        Vector2Int forwardDirection,
        out int immediateSafeContinuationCount,
        out int secondStepContinuationQuality)
    {
        BikeDirectionUtility.GetRelativeDirections(forwardDirection, out Vector2Int forward, out Vector2Int left, out Vector2Int right);

        immediateSafeContinuationCount = 0;
        secondStepContinuationQuality = 0;

        // Check the three relative follow-up moves from the candidate cell.
        EvaluateSecondStepDirection(originCell, forward, ref immediateSafeContinuationCount, ref secondStepContinuationQuality);
        EvaluateSecondStepDirection(originCell, left, ref immediateSafeContinuationCount, ref secondStepContinuationQuality);
        EvaluateSecondStepDirection(originCell, right, ref immediateSafeContinuationCount, ref secondStepContinuationQuality);
    }

    private void EvaluateSecondStepDirection(
        Vector2Int originCell,
        Vector2Int direction,
        ref int immediateSafeContinuationCount,
        ref int secondStepContinuationQuality)
    {
        Vector2Int nextCell = originCell + direction;
        if (_bike.IsCellBlocked(nextCell))
        {
            return;
        }

        // A legal second step counts once for immediate safety, then again for how open it stays afterwards.
        immediateSafeContinuationCount++;
        secondStepContinuationQuality += CountSafeContinuationMoves(nextCell, direction);
    }

    private int CountSafeContinuationMoves(Vector2Int originCell, Vector2Int forwardDirection)
    {
        BikeDirectionUtility.GetRelativeDirections(forwardDirection, out Vector2Int forward, out Vector2Int left, out Vector2Int right);

        int safeMoveCount = 0;
        if (!_bike.IsCellBlocked(originCell + forward))
        {
            safeMoveCount++;
        }

        if (!_bike.IsCellBlocked(originCell + left))
        {
            safeMoveCount++;
        }

        if (!_bike.IsCellBlocked(originCell + right))
        {
            safeMoveCount++;
        }

        return safeMoveCount;
    }

    // When safety is tied, lean slightly toward cells nearer the arena centre.
    private int CalculateCenterPreference(Vector2Int cell)
    {
        Vector2Int minCell = _bike.MinCell;
        Vector2Int maxCell = _bike.MaxCell;

        int centerXTwice = minCell.x + maxCell.x;
        int centerYTwice = minCell.y + maxCell.y;
        int distanceFromCenterTwice = Mathf.Abs((cell.x * 2) - centerXTwice) + Mathf.Abs((cell.y * 2) - centerYTwice);
        int maxDistanceFromCenterTwice = Mathf.Max(1, (maxCell.x - minCell.x) + (maxCell.y - minCell.y));

        return Mathf.Max(0, maxDistanceFromCenterTwice - distanceFromCenterTwice);
    }

    private static bool IsBetterHeuristicEvaluation(
        HeuristicDirectionEvaluation candidate,
        HeuristicDirectionEvaluation currentBest)
    {
        // Compare heuristic candidates in priority order: immediate survival, follow-up safety,
        // centre preference, forward bias, then a fixed tie-break.
        if (candidate.ImmediateSafeContinuationCount != currentBest.ImmediateSafeContinuationCount)
        {
            return candidate.ImmediateSafeContinuationCount > currentBest.ImmediateSafeContinuationCount;
        }

        if (candidate.SecondStepContinuationQuality != currentBest.SecondStepContinuationQuality)
        {
            return candidate.SecondStepContinuationQuality > currentBest.SecondStepContinuationQuality;
        }

        if (candidate.CenterPreference != currentBest.CenterPreference)
        {
            return candidate.CenterPreference > currentBest.CenterPreference;
        }

        if (candidate.ForwardPreference != currentBest.ForwardPreference)
        {
            return candidate.ForwardPreference > currentBest.ForwardPreference;
        }

        return candidate.TieBreakPriority < currentBest.TieBreakPriority;
    }

    private EpisodeSummary BuildEpisodeSummary(EpisodeOutcome outcome)
    {
        // Capture the final counters before EndEpisode clears the current episode state.
        return new EpisodeSummary(outcome, StepCount, GetCumulativeReward());
    }

    private void CacheReferences()
    {
        // Awake, OnEnable, Initialize, and OnEpisodeBegin can all arrive while the component is coming online,
        // so cache lazily instead of assuming one entry point always ran first.
        if (_bike == null)
        {
            _bike = GetComponent<PlayerBikeController>();
        }

        if (_behaviorParameters == null)
        {
            _behaviorParameters = GetComponent<BehaviorParameters>();
        }

        if (_rewardManager == null)
        {
            _rewardManager = GetComponent<RewardManager>();
        }
    }

    private void HandleBikeStepAdvanced(PlayerBikeController steppedBike)
    {
        if (!isActiveAndEnabled || steppedBike == null || steppedBike != _bike || _rewardManager == null)
        {
            return;
        }

        // Add the shaping reward after the shared bike step has fully finished.
        AddReward(_rewardManager.EvaluateStepReward(_bike));
    }

    private void SubscribeToBikeStepEvents()
    {
        if (_isSubscribedToBikeSteps || _bike == null)
        {
            return;
        }

        // The reward signal is added after each shared bike step, so this event is the sync point.
        _bike.StepAdvanced += HandleBikeStepAdvanced;
        _isSubscribedToBikeSteps = true;
    }

    private void UnsubscribeFromBikeStepEvents()
    {
        if (!_isSubscribedToBikeSteps || _bike == null)
        {
            return;
        }

        _bike.StepAdvanced -= HandleBikeStepAdvanced;
        _isSubscribedToBikeSteps = false;
    }

    private bool ValidateInferenceModelAssigned()
    {
        if (_behaviorParameters == null)
        {
            _behaviorParameters = GetComponent<BehaviorParameters>();
        }

        // Inference-only agents must already have a model in the scene, while training agents can start without one.
        if (_behaviorParameters != null &&
            _behaviorParameters.BehaviorType == BehaviorType.InferenceOnly &&
            _behaviorParameters.Model == null)
        {
            // Disable the component so the scene keeps running while still surfacing a clear configuration error.
            Debug.LogError("[LightBikeAgent] InferenceOnly but no model assigned. Disabling agent.");
            enabled = false;
            return false;
        }

        return true;
    }

    // Collapse large signed offsets into a simple left/right or ahead/behind hint for the policy.
    private static float CoarseSign(int value)
    {
        // Report only the coarse direction of the opponent rather than the exact offset.
        if (value > 1)
        {
            return 1f;
        }

        if (value < -1)
        {
            return -1f;
        }

        return 0f;
    }

    private float GetOpponentDistanceNormalization()
    {
        if (_bike == null)
        {
            return 1f;
        }

        // Normalise against the largest Manhattan distance available in the current arena.
        Vector2Int minCell = _bike.MinCell;
        Vector2Int maxCell = _bike.MaxCell;
        int maxManhattanDistance = (maxCell.x - minCell.x) + (maxCell.y - minCell.y);
        return Mathf.Max(1f, maxManhattanDistance);
    }

    private static void AddEmptyObservations(VectorSensor sensor)
    {
        // Keep the observation size stable even if the bike reference is missing during setup/teardown.
        for (int i = 0; i < ObservationCount; i++)
        {
            sensor.AddObservation(0f);
        }
    }
}
