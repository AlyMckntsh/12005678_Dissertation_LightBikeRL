using System;
using System.Collections.Generic;
using UnityEngine;
using Object = UnityEngine.Object;

/// <summary>
/// Holds the shared state for one arena. It keeps track of bike positions, occupied trail cells,
/// trail roots, and full round resets.
/// This is the source of truth for what cells are occupied and where each bike head currently is.
/// </summary>
[DisallowMultipleComponent]
public class ArenaContext : MonoBehaviour
{
    [Header("Arena")]
    [SerializeField] private Transform arenaOrigin;
    [SerializeField] private Transform learnerTrailRoot;
    [SerializeField] private Transform opponentTrailRoot;

    [Header("Bounds")]
    [SerializeField] private Vector2Int minCell = new Vector2Int(-8, -8);
    [SerializeField] private Vector2Int maxCell = new Vector2Int(8, 8);

    // These collections separate moving bike heads from permanent trail occupancy.
    private readonly List<PlayerBikeController> _activeBikes = new List<PlayerBikeController>(2);
    private readonly HashSet<Vector2Int> _occupiedTrailCells = new HashSet<Vector2Int>();
    private readonly Dictionary<PlayerBikeController, Vector2Int> _bikeCells = new Dictionary<PlayerBikeController, Vector2Int>();
    private bool _isRoundResetting;

    // MatchController uses this to know exactly when an arena is ready for the next opening decision request.
    internal event Action<ArenaContext> RoundResetCompleted;

    internal Transform ArenaOrigin
    {
        get
        {
            if (arenaOrigin != null)
            {
                return arenaOrigin;
            }

            // If no explicit origin is assigned, use the ArenaContext object's own transform.
            return transform;
        }
    }

    internal Vector2Int MinCell
    {
        get
        {
            // Normalise the inspector values so reversed min/max input still produces valid bounds.
            return new Vector2Int(
                Mathf.Min(minCell.x, maxCell.x),
                Mathf.Min(minCell.y, maxCell.y));
        }
    }

    internal Vector2Int MaxCell
    {
        get
        {
            // Normalise the inspector values so reversed min/max input still produces valid bounds.
            return new Vector2Int(
                Mathf.Max(minCell.x, maxCell.x),
                Mathf.Max(minCell.y, maxCell.y));
        }
    }

    internal Vector3 CellToWorld(Vector2Int cell, float cellSize, float y)
    {
        float safeCellSize = Mathf.Max(0.0001f, cellSize);
        Vector3 origin = ArenaOrigin.position;

        // Convert logical grid coordinates into one shared arena space.
        return new Vector3(
            origin.x + (cell.x * safeCellSize),
            y,
            origin.z + (cell.y * safeCellSize));
    }

    internal void RegisterBike(PlayerBikeController bike)
    {
        if (bike == null)
        {
            return;
        }

        // Bikes register on enable so resets and occupancy checks can iterate the active arena set.
        if (!_activeBikes.Contains(bike))
        {
            _activeBikes.Add(bike);
        }

        _bikeCells[bike] = bike.CurrentCell;
    }

    internal void UnregisterBike(PlayerBikeController bike)
    {
        if (bike == null)
        {
            return;
        }

        // Remove disabled bikes so collision checks and resets only touch the active arena set.
        _activeBikes.Remove(bike);
        _bikeCells.Remove(bike);
    }

    internal void CopyActiveBikesTo(List<PlayerBikeController> results)
    {
        if (results == null)
        {
            return;
        }

        // Copy into a caller-owned list so simulation code can work with a stable snapshot.
        results.Clear();
        for (int i = 0; i < _activeBikes.Count; i++)
        {
            PlayerBikeController bike = _activeBikes[i];
            if (bike != null)
            {
                results.Add(bike);
            }
        }
    }

    internal PlayerBikeController FindOpponent(PlayerBikeController self)
    {
        // Each arena expects one learner and one opponent, so "the other registered bike" is the opponent.
        for (int i = 0; i < _activeBikes.Count; i++)
        {
            PlayerBikeController bike = _activeBikes[i];
            if (bike != null && bike != self)
            {
                return bike;
            }
        }

        return null;
    }

    internal PlayerBikeController.CollisionReason ResolveCollisionReason(PlayerBikeController requester, Vector2Int targetCell)
    {
        // requester is ignored when checking bike heads so a bike does not collide with its own current head cell.
        // Trail collisions are checked first because trail stays fixed for the whole round.
        if (_occupiedTrailCells.Contains(targetCell))
        {
            return PlayerBikeController.CollisionReason.Trail;
        }

        // Then check whether another bike head is already on that cell.
        if (IsOccupiedByOtherBike(requester, targetCell))
        {
            return PlayerBikeController.CollisionReason.OpponentHead;
        }

        return PlayerBikeController.CollisionReason.None;
    }

    internal void MarkTrailCellOccupied(Vector2Int cell)
    {
        // Bikes call this when they leave a cell so that cell becomes a permanent trail obstacle for the round.
        _occupiedTrailCells.Add(cell);
    }

    internal void UpdateBikeCell(PlayerBikeController bike, Vector2Int cell)
    {
        if (bike == null)
        {
            return;
        }

        // Keep the live head-position map in sync after moves and resets.
        _bikeCells[bike] = cell;
    }

    internal Transform GetTrailRoot(bool isOpponent)
    {
        if (isOpponent)
        {
            return opponentTrailRoot;
        }

        // Keep learner and opponent trail visuals on separate parents for easier scene inspection.
        return learnerTrailRoot;
    }

    internal void TriggerRoundReset()
    {
        if (_isRoundResetting)
        {
            return;
        }

        // Centralise the reset so trails, occupancy, and bike transforms all return to start state together.
        // Clear the old arena first, then ask every registered bike to reset itself.
        _isRoundResetting = true;
        ClearArenaTrailsAndState();
        ResetAllRegisteredBikes();
        _isRoundResetting = false;

        if (RoundResetCompleted != null)
        {
            // MatchController uses this to request the first decisions only after both bikes
            // have finished resetting for the next round.
            // Fire this after _isRoundResetting is false so listeners see the arena as ready.
            RoundResetCompleted.Invoke(this);
        }
    }

    private void ClearArenaTrailsAndState()
    {
        ReturnBikeTrailSegmentsToPool();

        // Clear the old round first so the bikes can safely place themselves again during reset.
        _occupiedTrailCells.Clear();
        _bikeCells.Clear();
        ClearTrailRoot(learnerTrailRoot);
        ClearTrailRoot(opponentTrailRoot);
    }

    private void ReturnBikeTrailSegmentsToPool()
    {
        // Let each bike release its own pooled trail pieces before the arena clears child objects.
        for (int i = 0; i < _activeBikes.Count; i++)
        {
            PlayerBikeController bike = _activeBikes[i];
            if (bike != null)
            {
                bike.ClearTrailSegments();
            }
        }
    }

    private bool IsOccupiedByOtherBike(PlayerBikeController requester, Vector2Int targetCell)
    {
        // This only checks live head positions. Permanent trail occupancy is handled by _occupiedTrailCells.
        foreach (KeyValuePair<PlayerBikeController, Vector2Int> pair in _bikeCells)
        {
            PlayerBikeController bike = pair.Key;
            if (bike == null || bike == requester)
            {
                continue;
            }

            if (pair.Value == targetCell)
            {
                return true;
            }
        }

        return false;
    }

    private void ResetAllRegisteredBikes()
    {
        // Ask every bike to rebuild its own local state after the arena-wide clear.
        for (int i = 0; i < _activeBikes.Count; i++)
        {
            PlayerBikeController bike = _activeBikes[i];
            if (bike != null)
            {
                bike.ResetForRound();
            }
        }
    }

    private static void ClearTrailRoot(Transform root)
    {
        if (root == null)
        {
            return;
        }

        // Remove any leftover child objects that are still sitting under the visible trail roots.
        for (int i = root.childCount - 1; i >= 0; i--)
        {
            Transform child = root.GetChild(i);
            if (child != null)
            {
                Object.Destroy(child.gameObject);
            }
        }
    }
}
