using UnityEngine;

/// <summary>
/// Small helper methods for turning and comparing grid directions.
/// </summary>
internal static class BikeDirectionUtility
{
    internal static Vector2Int GetForwardDirection(Vector2Int direction)
    {
        // Treat a zero direction as "up" so reset/setup code always has a valid forward vector.
        if (direction == Vector2Int.zero)
        {
            return Vector2Int.up;
        }

        return direction;
    }

    internal static void GetRelativeDirections(
        Vector2Int direction,
        out Vector2Int forward,
        out Vector2Int left,
        out Vector2Int right)
    {
        forward = GetForwardDirection(direction);
        // Rotate the forward vector by 90 degrees to derive local left and right.
        left = new Vector2Int(-forward.y, forward.x);
        right = new Vector2Int(forward.y, -forward.x);
    }

    internal static Vector2Int Rotate(Vector2Int direction, int turn)
    {
        Vector2Int forward = GetForwardDirection(direction);

        // Negative turn means left, positive turn means right, and zero keeps the same heading.
        if (turn < 0)
        {
            return new Vector2Int(-forward.y, forward.x);
        }

        if (turn > 0)
        {
            return new Vector2Int(forward.y, -forward.x);
        }

        return forward;
    }
}
