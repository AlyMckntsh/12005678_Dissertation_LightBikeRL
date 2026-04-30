using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Calculates the fixed dissertation reward conditions used by the light bike agents.
/// Each shared step is turned into a board snapshot, then compared against the previous one
/// so the shaping reward reflects progress instead of raw absolute values.
/// </summary>
public class RewardManager : MonoBehaviour
{
    /// <summary>
    /// Selects which fixed reward formula the run should use.
    /// </summary>
    public enum RewardCondition
    {
        SurvivalOnly = 0,
        ArenaControl = 1
    }

    /// <summary>
    /// Captures the board features used to score one shared step.
    /// </summary>
    private struct RewardSnapshot
    {
        internal RewardSnapshot(
            float reachableSpaceDifferenceNormalized,
            int selfSafeMoves,
            float selfSafeMoveScore,
            float centerControlNormalized)
        {
            ReachableSpaceDifferenceNormalized = reachableSpaceDifferenceNormalized;
            SelfSafeMoves = selfSafeMoves;
            SelfSafeMoveScore = selfSafeMoveScore;
            CenterControlNormalized = centerControlNormalized;
        }

        internal readonly float ReachableSpaceDifferenceNormalized;
        internal readonly int SelfSafeMoves;
        internal readonly float SelfSafeMoveScore;
        internal readonly float CenterControlNormalized;
    }

    // These tuning values stay fixed so training and evaluation runs remain directly comparable.
    // They define the dissertation reward formula rather than being learned from the environment.
    private const int RewardTuningReferenceMaxStep = 1500;
    private const int ReachabilityBufferCapacity = 1024;
    private const int MaxSafeMoveCount = 3;

    private const float WinReward = 1f;
    private const float LossPenalty = -1f;
    private const float DrawReward = 0f;
    private const float SurvivalRewardPerStep = 0.45f / RewardTuningReferenceMaxStep;
    private const float ReachableSpaceDeltaReward = 0.06f;
    private const float SafeMoveDeltaReward = 0.03f;
    private const float ZeroSafeMovePenalty = -0.03f;
    private const float OneSafeMovePenalty = -0.015f;
    private const float TwoSafeMovePenalty = 0f;
    private const float ArenaControlCenterReward = 0.0002f;

    // Reuse the flood-fill buffers every step so reward calculation does not allocate during training.
    private readonly Queue<Vector2Int> _searchQueue = new Queue<Vector2Int>(ReachabilityBufferCapacity);
    private readonly HashSet<Vector2Int> _visitedCells = new HashSet<Vector2Int>(ReachabilityBufferCapacity);

    private RewardCondition _rewardCondition = RewardCondition.SurvivalOnly;
    private bool _hasPreviousSnapshot;
    private RewardSnapshot _lastSnapshot;

    internal void SetRewardCondition(RewardCondition rewardCondition)
    {
        if (_rewardCondition == rewardCondition)
        {
            return;
        }

        // Switching reward formulas invalidates any step-to-step delta from the previous mode.
        _rewardCondition = rewardCondition;
        InvalidateStepSnapshot();
    }

    internal void BeginEpisode(PlayerBikeController bike)
    {
        // Take the opening snapshot now so the first live step has a baseline to compare against.
        _hasPreviousSnapshot = BuildRewardSnapshot(bike, out _lastSnapshot);
    }

    internal float EvaluateStepReward(PlayerBikeController bike)
    {
        if (bike == null || bike.IsDead)
        {
            return 0f;
        }

        // Build the current board snapshot first. If that fails, there is no valid shaping reward this step.
        if (!BuildRewardSnapshot(bike, out RewardSnapshot currentSnapshot))
        {
            InvalidateStepSnapshot();
            return 0f;
        }

        float reward;
        switch (_rewardCondition)
        {
            case RewardCondition.ArenaControl:
                // ArenaControl adds a small centre-position bonus on top of the shared survival shaping.
                reward = EvaluateArenaControlStepReward(currentSnapshot);
                break;

            default:
                // SurvivalOnly keeps the shaping focused on staying alive and preserving future space.
                reward = EvaluateSurvivalOnlyStepReward(currentSnapshot);
                break;
        }

        _lastSnapshot = currentSnapshot;
        _hasPreviousSnapshot = true;
        return reward;
    }

    internal float EvaluateTerminalReward(LightBikeAgent.EpisodeOutcome outcome)
    {
        // Terminal rewards stay separate from shaping so the win/loss signal is always explicit.
        switch (outcome)
        {
            case LightBikeAgent.EpisodeOutcome.Win:
                return WinReward;

            case LightBikeAgent.EpisodeOutcome.Draw:
                return DrawReward;

            default:
                return LossPenalty;
        }
    }

    private float EvaluateSurvivalOnlyStepReward(RewardSnapshot currentSnapshot)
    {
        float reward = EvaluateSharedStepReward(currentSnapshot);
        reward += EvaluateDeltaRewards(currentSnapshot, ReachableSpaceDeltaReward);
        return reward;
    }

    private float EvaluateArenaControlStepReward(RewardSnapshot currentSnapshot)
    {
        float reward = EvaluateSharedStepReward(currentSnapshot);
        reward += ArenaControlCenterReward * currentSnapshot.CenterControlNormalized;
        reward += EvaluateDeltaRewards(currentSnapshot, ReachableSpaceDeltaReward);
        return reward;
    }

    private float EvaluateSharedStepReward(RewardSnapshot currentSnapshot)
    {
        // Both reward modes always pay the same survival baseline and dead-end pressure.
        return SurvivalRewardPerStep + EvaluateNearDeadEndPenalty(currentSnapshot.SelfSafeMoves);
    }

    private float EvaluateDeltaRewards(RewardSnapshot currentSnapshot, float reachableSpaceDeltaReward)
    {
        if (!_hasPreviousSnapshot)
        {
            return 0f;
        }

        // Reward changes from the previous step rather than stacking the same advantage forever.
        float reward = 0f;
        reward += reachableSpaceDeltaReward *
                  ClampSignedUnit(currentSnapshot.ReachableSpaceDifferenceNormalized - _lastSnapshot.ReachableSpaceDifferenceNormalized);
        reward += SafeMoveDeltaReward *
                  ClampSignedUnit(currentSnapshot.SelfSafeMoveScore - _lastSnapshot.SelfSafeMoveScore);
        return reward;
    }

    private static float EvaluateNearDeadEndPenalty(int selfSafeMoves)
    {
        // Fewer legal exits means the bike is closer to trapping itself, so the penalty increases.
        if (selfSafeMoves <= 0)
        {
            return ZeroSafeMovePenalty;
        }

        if (selfSafeMoves == 1)
        {
            return OneSafeMovePenalty;
        }

        if (selfSafeMoves == 2)
        {
            return TwoSafeMovePenalty;
        }

        return 0f;
    }

    private bool BuildRewardSnapshot(PlayerBikeController selfBike, out RewardSnapshot snapshot)
    {
        snapshot = default;

        if (selfBike == null || selfBike.IsDead)
        {
            return false;
        }

        PlayerBikeController opponent = selfBike.FindOpponent();
        if (opponent == null || opponent.IsDead)
        {
            // Shaping comparisons only make sense while both bikes are still active on the board.
            return false;
        }

        Vector2Int minCell = selfBike.MinCell;
        Vector2Int maxCell = selfBike.MaxCell;

        // Measure how much safe space each bike can still reach from the current board state.
        int totalCellCount = CalculateTotalCellCount(minCell, maxCell);
        int selfReachableCells = EstimateReachableCellCount(selfBike, selfBike.CurrentCell, minCell, maxCell);
        int opponentReachableCells = EstimateReachableCellCount(opponent, opponent.CurrentCell, minCell, maxCell);
        int selfSafeMoves = CountSafeMoves(selfBike);

        snapshot = new RewardSnapshot(
            reachableSpaceDifferenceNormalized: (selfReachableCells - opponentReachableCells) / (float)totalCellCount,
            selfSafeMoves: selfSafeMoves,
            selfSafeMoveScore: NormalizeSafeMoveCount(selfSafeMoves),
            centerControlNormalized: CalculateCenterControl(selfBike.CurrentCell, minCell, maxCell));

        return true;
    }

    private static int CalculateTotalCellCount(Vector2Int minCell, Vector2Int maxCell)
    {
        // Use total arena area to turn reachable-space differences into a size-independent ratio.
        return Mathf.Max(1, ((maxCell.x - minCell.x) + 1) * ((maxCell.y - minCell.y) + 1));
    }

    private int EstimateReachableCellCount(
        PlayerBikeController bike,
        Vector2Int origin,
        Vector2Int minCell,
        Vector2Int maxCell)
    {
        if (!IsWithinWindow(origin, minCell, maxCell) || bike.IsCellBlocked(origin))
        {
            return 0;
        }

        // Flood-fill every cell this bike can still reach without hitting a wall, trail, or head collision.
        _searchQueue.Clear();
        _visitedCells.Clear();
        _searchQueue.Enqueue(origin);
        _visitedCells.Add(origin);

        while (_searchQueue.Count > 0)
        {
            Vector2Int cell = _searchQueue.Dequeue();
            AddReachableNeighbor(bike, cell + Vector2Int.up, minCell, maxCell);
            AddReachableNeighbor(bike, cell + Vector2Int.down, minCell, maxCell);
            AddReachableNeighbor(bike, cell + Vector2Int.left, minCell, maxCell);
            AddReachableNeighbor(bike, cell + Vector2Int.right, minCell, maxCell);
        }

        return _visitedCells.Count;
    }

    private void AddReachableNeighbor(
        PlayerBikeController bike,
        Vector2Int neighbor,
        Vector2Int minCell,
        Vector2Int maxCell)
    {
        if (!IsWithinWindow(neighbor, minCell, maxCell))
        {
            return;
        }

        if (_visitedCells.Contains(neighbor) || bike.IsCellBlocked(neighbor))
        {
            return;
        }

        // Only enqueue cells that are still survivable and have not already been counted.
        _visitedCells.Add(neighbor);
        _searchQueue.Enqueue(neighbor);
    }

    private static bool IsWithinWindow(Vector2Int cell, Vector2Int minCell, Vector2Int maxCell)
    {
        return cell.x >= minCell.x &&
               cell.x <= maxCell.x &&
               cell.y >= minCell.y &&
               cell.y <= maxCell.y;
    }

    private static int CountSafeMoves(PlayerBikeController bike)
    {
        if (bike == null || bike.IsDead)
        {
            return 0;
        }

        // Count legal forward, left, and right moves from the bike's current local frame.
        BikeDirectionUtility.GetRelativeDirections(
            bike.CurrentDirection,
            out Vector2Int forward,
            out Vector2Int left,
            out Vector2Int right);

        int safeMoves = 0;
        if (!bike.IsDirectionBlocked(forward))
        {
            safeMoves++;
        }

        if (!bike.IsDirectionBlocked(left))
        {
            safeMoves++;
        }

        if (!bike.IsDirectionBlocked(right))
        {
            safeMoves++;
        }

        return safeMoves;
    }

    private static float NormalizeSafeMoveCount(int safeMoves)
    {
        // Convert the raw 0..3 safe-move count into a consistent 0..1 feature value.
        return Mathf.Clamp01(safeMoves / (float)MaxSafeMoveCount);
    }

    private static float CalculateCenterControl(Vector2Int cell, Vector2Int minCell, Vector2Int maxCell)
    {
        // Cells nearer the arena centre get a higher score because they usually leave more future options open.
        float centerX = (minCell.x + maxCell.x) * 0.5f;
        float centerY = (minCell.y + maxCell.y) * 0.5f;
        float distanceFromCenter = Mathf.Abs(cell.x - centerX) + Mathf.Abs(cell.y - centerY);

        float maxDistanceFromCenter = Mathf.Max(
            0.5f,
            ((maxCell.x - minCell.x) * 0.5f) + ((maxCell.y - minCell.y) * 0.5f));

        return 1f - Mathf.Clamp01(distanceFromCenter / maxDistanceFromCenter);
    }

    private static float ClampSignedUnit(float value)
    {
        // Cap one-step reward deltas so a single noisy state change cannot dominate the reward signal.
        return Mathf.Clamp(value, -1f, 1f);
    }

    private void InvalidateStepSnapshot()
    {
        // Reset the step-to-step baseline whenever continuity is broken by reset or mode change.
        _hasPreviousSnapshot = false;
        _lastSnapshot = default;
    }
}
