using System;
using UnityEngine;

/// <summary>
/// Controls one bike on the grid. It stores the next turn chosen by the agent,
/// moves one cell at a time, leaves a trail behind, and reports crashes.
/// EpisodeManager decides when the shared step happens; this class only prepares and commits this bike's part of it.
/// </summary>
[DisallowMultipleComponent]
public class PlayerBikeController : MonoBehaviour
{
    /// <summary>
    /// Records why a move ended in a crash.
    /// </summary>
    internal enum CollisionReason
    {
        None = 0,
        Wall = 1,
        Trail = 2,
        OpponentHead = 3,
        SameCellContest = 4,
        HeadSwap = 5
    }

    [Header("Movement")]
    [SerializeField] private float cellSize = 1f;
    [SerializeField] private float stepDuration = 0.15f;
    [SerializeField] private Vector2Int startCell = Vector2Int.zero;
    [SerializeField] private Vector2Int startDirection = Vector2Int.up;

    [Header("Trail")]
    [SerializeField] private float trailHeight = 1f;
    [SerializeField] private Material trailMaterial;
    // This flag marks the bike as the opponent role for trail parenting and winner-resolution logic.
    [SerializeField] private bool isOpponent;

    [Header("References")]
    [SerializeField] private ArenaContext arenaContext;
    [SerializeField] private EpisodeManager episodeManager;

    private Vector2Int _currentCell;
    private Vector2Int _currentDirection;
    // _pendingTurn stores the latest agent input this frame, while _queuedTurn is the turn that
    // will actually be consumed on the next shared grid step.
    private int _queuedTurn;
    private readonly BikeTrailView _trailView = new BikeTrailView();
    private bool _isDead;
    private bool _hasPendingTurn;
    private int _pendingTurn;

    internal Vector2Int CurrentCell
    {
        get { return _currentCell; }
    }

    internal Vector2Int CurrentDirection
    {
        get { return _currentDirection; }
    }

    internal bool IsDead
    {
        get { return _isDead; }
    }

    internal Vector2Int StartCell
    {
        get { return startCell; }
    }

    internal Vector2Int EffectiveStartDirection
    {
        get { return BikeDirectionUtility.GetForwardDirection(startDirection); }
    }

    internal Vector2Int MinCell
    {
        get { return GetArenaMinCell(); }
    }

    internal Vector2Int MaxCell
    {
        get { return GetArenaMaxCell(); }
    }

    internal float StepDuration
    {
        get { return stepDuration; }
    }

    internal ArenaContext ArenaContext
    {
        get { return arenaContext; }
    }

    internal bool IsOpponentRole
    {
        get { return isOpponent; }
    }

    internal event Action<PlayerBikeController> StepAdvanced;

    private void OnEnable()
    {
        if (arenaContext != null)
        {
            // Register early so resets and collision checks can include this bike as soon as it becomes active.
            arenaContext.RegisterBike(this);
        }
    }

    private void OnDisable()
    {
        // Clear pooled visuals when the bike leaves play so the arena does not keep stale trail state around.
        ClearTrailSegments();

        if (arenaContext != null)
        {
            arenaContext.UnregisterBike(this);
        }
    }

    private void Start()
    {
        if (startDirection == Vector2Int.zero)
        {
            startDirection = Vector2Int.up;
        }

        // Normalise the setup, validate scene wiring, then enter the opening round state.
        if (!ValidateReferences())
        {
            enabled = false;
            return;
        }

        ConfigurePhysicsComponents();
        RefreshTrailViewConfiguration();
        ResetBikeState();
    }

    private bool ValidateReferences()
    {
        // Fail early on missing arena wiring because movement, trail drawing, and resets all depend on it.
        if (arenaContext == null)
        {
            Debug.LogError("[PlayerBikeController] ArenaContext reference is required.", this);
            return false;
        }

        if (episodeManager == null)
        {
            Debug.LogError("[PlayerBikeController] EpisodeManager reference is required.", this);
            return false;
        }

        if (ResolveTrailRoot() == null)
        {
            Debug.LogError("[PlayerBikeController] ArenaContext trail root is not assigned for this bike role.", this);
            return false;
        }

        return true;
    }

    internal Vector2Int PrepareStepIntent()
    {
        // First copy in any turn that the agent has submitted since the last shared step.
        CommitPendingTurn();

        // Apply the chosen turn once, then work out the next cell in front of the bike.
        ApplyQueuedTurnIfNeeded();
        return _currentCell + _currentDirection;
    }

    internal CollisionReason ResolveBaseCollisionReason(Vector2Int nextCell)
    {
        // EpisodeManager uses this before bike-versus-bike conflict checks are applied.
        return WillDieAtCell(nextCell);
    }

    internal void CommitResolvedMove(Vector2Int nextCell)
    {
        // Leave the current cell as trail first, then move the head into the next cell.
        LeaveTrailAtCurrentCell();
        MoveToCell(nextCell);
    }

    internal void NotifyStepAdvanced()
    {
        if (StepAdvanced != null)
        {
            // LightBikeAgent and MatchController listen here to add reward and request the next decisions.
            StepAdvanced.Invoke(this);
        }
    }

    internal void CommitResolvedCrash(CollisionReason collisionReason)
    {
        // A failed move still leaves the current head position behind as trail.
        LeaveTrailAtCurrentCell();
        transform.position = CellToWorld(_currentCell);

        CollisionReason resolvedReason = collisionReason;
        if (resolvedReason == CollisionReason.None)
        {
            // Fall back to a generic blocked-cell crash if the caller did not pass a more specific reason.
            resolvedReason = CollisionReason.Trail;
        }

        Die(resolvedReason);
    }

    internal void SubmitTurn(int turn)
    {
        // The agent may submit more than once before the next shared step. The latest request wins.
        _hasPendingTurn = true;
        _pendingTurn = Mathf.Clamp(turn, -1, 1);
    }

    internal void ClearPendingTurn()
    {
        // Clearing both buffers prevents an old turn from leaking into the next round.
        _queuedTurn = 0;
        _hasPendingTurn = false;
        _pendingTurn = 0;
    }

    internal bool IsDirectionBlocked(Vector2Int direction)
    {
        if (direction == Vector2Int.zero)
        {
            return true;
        }

        // Convert a direction check into the target cell check used by observations, heuristics, and rewards.
        return IsCellBlocked(_currentCell + direction);
    }

    internal bool IsCellBlocked(Vector2Int cell)
    {
        // Shared cell-level collision query used by observations, heuristics, and reward estimation.
        return WillDieAtCell(cell) != CollisionReason.None;
    }

    internal PlayerBikeController FindOpponent()
    {
        if (arenaContext == null)
        {
            return null;
        }

        // ArenaContext owns the active bike list, so it is the authoritative place to ask for the opponent.
        return arenaContext.FindOpponent(this);
    }

    private void ApplyFacing()
    {
        // Keep the visible bike aligned with the logical grid direction.
        Vector3 forward = new Vector3(_currentDirection.x, 0f, _currentDirection.y);
        if (forward.sqrMagnitude > 0f)
        {
            transform.rotation = Quaternion.LookRotation(forward, Vector3.up);
        }
    }

    private void CommitPendingTurn()
    {
        if (!_hasPendingTurn)
        {
            return;
        }

        // The latest submitted turn wins for this frame.
        _queuedTurn = _pendingTurn;
        _hasPendingTurn = false;
    }

    private void ApplyQueuedTurnIfNeeded()
    {
        if (_queuedTurn == 0)
        {
            return;
        }

        // Consume the buffered turn exactly once on the next shared step.
        _currentDirection = BikeDirectionUtility.Rotate(_currentDirection, _queuedTurn);
        _queuedTurn = 0;
        ApplyFacing();
    }

    private CollisionReason WillDieAtCell(Vector2Int nextCell)
    {
        // Check walls first, then ask the arena about trail and head occupancy.
        if (IsOutOfBounds(nextCell))
        {
            return CollisionReason.Wall;
        }

        return GetOccupancyCollisionReason(nextCell);
    }

    private bool IsOutOfBounds(Vector2Int cell)
    {
        // Pull the playable window from ArenaContext so every bike in the arena agrees on the same limits.
        Vector2Int minCell = GetArenaMinCell();
        Vector2Int maxCell = GetArenaMaxCell();

        return cell.x < minCell.x ||
               cell.x > maxCell.x ||
               cell.y < minCell.y ||
               cell.y > maxCell.y;
    }

    private Vector2Int GetArenaMinCell()
    {
        return arenaContext.MinCell;
    }

    private Vector2Int GetArenaMaxCell()
    {
        return arenaContext.MaxCell;
    }

    private void LeaveTrailAtCurrentCell()
    {
        // Trail is both a visible wall and a logical occupied cell, so update both in one place.
        _trailView.SpawnTrailSegment(_currentCell, CellToWorld(_currentCell));

        if (arenaContext != null)
        {
            arenaContext.MarkTrailCellOccupied(_currentCell);
        }
    }

    private void MoveToCell(Vector2Int nextCell)
    {
        // Update the bike state first, then tell the arena where the new head cell is.
        _currentCell = nextCell;

        if (arenaContext != null)
        {
            arenaContext.UpdateBikeCell(this, _currentCell);
        }

        transform.position = CellToWorld(nextCell);
    }

    private Vector3 CellToWorld(Vector2Int cell)
    {
        if (arenaContext != null)
        {
            return arenaContext.CellToWorld(cell, cellSize, transform.position.y);
        }

        // Fallback for setup errors so the bike can still be positioned in local grid space.
        return new Vector3(cell.x * cellSize, transform.position.y, cell.y * cellSize);
    }

    private Transform ResolveTrailRoot()
    {
        if (arenaContext == null)
        {
            return null;
        }

        // Learner and opponent keep separate trail parents so the hierarchy stays easy to read.
        return arenaContext.GetTrailRoot(isOpponent);
    }

    private void RefreshTrailViewConfiguration()
    {
        // Push the latest inspector values into the pooled trail view helper.
        _trailView.Configure(transform, ResolveTrailRoot(), cellSize, trailHeight, trailMaterial);
    }

    private void ConfigurePhysicsComponents()
    {
        // Movement is grid-driven, not physics-driven, so keep the rigidbody passive and collider trigger-only.
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.isKinematic = true;
            rb.useGravity = false;
        }

        BoxCollider bikeCollider = GetComponent<BoxCollider>();
        if (bikeCollider != null)
        {
            bikeCollider.isTrigger = true;
            bikeCollider.center = Vector3.zero;
            bikeCollider.size = new Vector3(cellSize * 0.9f, trailHeight, cellSize * 0.9f);
        }
    }

    private CollisionReason GetOccupancyCollisionReason(Vector2Int targetCell)
    {
        if (arenaContext == null)
        {
            return CollisionReason.None;
        }

        // ArenaContext resolves trail hits and head occupancy because it has the shared arena state.
        return arenaContext.ResolveCollisionReason(this, targetCell);
    }

    internal void ClearTrailSegments()
    {
        // Keep the trail view helper behind one method so arena reset code never has to know its internals.
        _trailView.ClearTrailSegments();
    }

    private void Die(CollisionReason collisionReason)
    {
        if (_isDead)
        {
            return;
        }

        _isDead = true;

        // The collision reason has already done its job by the time we get here.
        // This method now just flips the bike into its dead state and informs the round controller.
        // EpisodeManager owns the round result, so report the death there instead of ending the episode locally.
        if (episodeManager != null)
        {
            episodeManager.HandleBikeDeath(this);
        }
    }

    private void ResetBikeState()
    {
        // This runs on Start and on every round reset.
        // Clear out the old round first.
        _isDead = false;
        ClearPendingTurn();
        ClearTrailSegments();

        // Put the bike back on its starting cell and direction.
        _currentCell = startCell;
        _currentDirection = EffectiveStartDirection;

        // Then update the transform and arena so both match the reset state.
        transform.position = CellToWorld(_currentCell);

        if (arenaContext != null)
        {
            arenaContext.UpdateBikeCell(this, _currentCell);
        }

        ApplyFacing();
        RefreshTrailViewConfiguration();
        // Re-enable the bike after a reset so it can participate in the next shared step.
        enabled = true;
    }

    internal void ResetForRound()
    {
        // ArenaContext calls this after it clears the shared arena state.
        ResetBikeState();
    }
}
