using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Pool;
using Object = UnityEngine.Object;

/// <summary>
/// Handles the visual trail pieces for one bike. It only deals with pooled cube objects and does not
/// make any gameplay decisions.
/// Using ObjectPool keeps the trail visuals cheap even when long rounds spawn many segments.
/// </summary>
internal sealed class BikeTrailView
{
    private const string PoolRootName = "__TrailPool";
    private const int DefaultPoolCapacity = 128;
    private const int MaxPoolSize = 4096;
    private const float MinTrailDimension = 0.05f;

    // Track only the trail pieces currently visible in the arena so they can all be released together on reset.
    private readonly List<GameObject> _activeSegments = new List<GameObject>(256);

    private Transform _ownerTransform;
    private Transform _trailRoot;
    private Transform _trailPoolRoot;
    private ObjectPool<GameObject> _trailSegmentPool;
    private Material _trailMaterial;
    private float _cellSize = 1f;
    private float _trailHeight = 1f;

    internal void Configure(
        Transform ownerTransform,
        Transform trailRoot,
        float cellSize,
        float trailHeight,
        Material trailMaterial)
    {
        // Called on start and reset so the trail visuals follow the bike's current inspector settings.
        _ownerTransform = ownerTransform;
        _trailRoot = trailRoot;
        _trailMaterial = trailMaterial;
        _cellSize = Mathf.Max(MinTrailDimension, cellSize);
        _trailHeight = Mathf.Max(MinTrailDimension, trailHeight);
    }

    internal void SpawnTrailSegment(Vector2Int cell, Vector3 worldPosition)
    {
        if (_ownerTransform == null || _trailRoot == null)
        {
            return;
        }

        // Pull one cube from the pool, then place and size it for this grid cell.
        EnsurePool();
        if (_trailSegmentPool == null)
        {
            return;
        }

        GameObject segment = _trailSegmentPool.Get();
        segment.name = $"Trail_{cell.x}_{cell.y}";
        segment.transform.SetParent(_trailRoot, false);
        segment.transform.position = worldPosition;
        segment.transform.localScale = new Vector3(_cellSize, _trailHeight, _cellSize);

        Renderer renderer = segment.GetComponent<Renderer>();
        if (renderer != null && _trailMaterial != null)
        {
            renderer.sharedMaterial = _trailMaterial;
        }

        _activeSegments.Add(segment);
    }

    internal void ClearTrailSegments()
    {
        if (_activeSegments.Count <= 0)
        {
            return;
        }

        EnsurePool();
        if (_trailSegmentPool == null)
        {
            _activeSegments.Clear();
            return;
        }

        // When a round ends, return every live segment to the pool so it can be reused next time.
        for (int i = _activeSegments.Count - 1; i >= 0; i--)
        {
            GameObject segment = _activeSegments[i];
            if (segment != null)
            {
                _trailSegmentPool.Release(segment);
            }
        }

        _activeSegments.Clear();
    }

    private void EnsurePool()
    {
        if (_trailSegmentPool != null || _ownerTransform == null)
        {
            return;
        }

        // Build the pool the first time this bike needs to draw trail.
        EnsurePoolRoot();
        if (_trailPoolRoot == null)
        {
            return;
        }

        _trailSegmentPool = new ObjectPool<GameObject>(
            CreateTrailSegment,
            OnGetTrailSegment,
            OnReleaseTrailSegment,
            OnDestroyTrailSegment,
            collectionCheck: false,
            defaultCapacity: DefaultPoolCapacity,
            maxSize: MaxPoolSize);
    }

    private void EnsurePoolRoot()
    {
        if (_trailPoolRoot != null || _ownerTransform == null)
        {
            return;
        }

        Transform existingPoolRoot = _ownerTransform.Find(PoolRootName);
        if (existingPoolRoot != null)
        {
            _trailPoolRoot = existingPoolRoot;
            return;
        }

        // Keep the inactive pooled trail pieces under the bike so they stay tidy in the Hierarchy.
        GameObject poolRootObject = new GameObject(PoolRootName);
        poolRootObject.transform.SetParent(_ownerTransform, false);
        _trailPoolRoot = poolRootObject.transform;
    }

    private GameObject CreateTrailSegment()
    {
        // Trail cubes are visual markers only, so their collider is disabled immediately.
        GameObject segment = GameObject.CreatePrimitive(PrimitiveType.Cube);
        segment.name = "TrailSegment";
        segment.transform.SetParent(_trailPoolRoot, false);
        segment.SetActive(false);

        Collider segmentCollider = segment.GetComponent<Collider>();
        if (segmentCollider != null)
        {
            segmentCollider.enabled = false;
        }

        Renderer renderer = segment.GetComponent<Renderer>();
        if (renderer != null && _trailMaterial != null)
        {
            renderer.sharedMaterial = _trailMaterial;
        }

        return segment;
    }

    private static void OnGetTrailSegment(GameObject segment)
    {
        if (segment != null)
        {
            // Pool callback: reactivate the segment before it is placed into the live trail.
            segment.SetActive(true);
        }
    }

    private void OnReleaseTrailSegment(GameObject segment)
    {
        if (segment == null)
        {
            return;
        }

        // Return released pieces to the hidden pool root so the active trail hierarchy stays readable.
        if (_trailPoolRoot != null)
        {
            segment.transform.SetParent(_trailPoolRoot, false);
        }

        segment.SetActive(false);
    }

    private static void OnDestroyTrailSegment(GameObject segment)
    {
        if (segment != null)
        {
            // Pool callback: permanently destroy extra segments if the pool needs to shrink.
            Object.Destroy(segment);
        }
    }
}
