using Unity.MLAgents;

/// <summary>
/// Sends simple learner episode statistics to the ML-Agents stats recorder.
/// These show up as custom scalar series alongside the trainer's built-in metrics.
/// </summary>
internal sealed class TrainingStatsReporter
{
    // Group the custom learner metrics under one TensorBoard-friendly prefix.
    private const string MetricPrefix = "LightBike/Training";

    internal void RecordEpisode(LightBikeAgent.EpisodeSummary summary)
    {
        StatsRecorder statsRecorder = Academy.Instance.StatsRecorder;
        if (statsRecorder == null)
        {
            return;
        }

        // Convert the categorical outcome into a scalar so TensorBoard can average it into a win-rate series.
        float winValue = summary.DidWin ? 1f : 0f;

        // Write one set of learner stats after each finished episode.
        statsRecorder.Add($"{MetricPrefix}/WinRate", winValue, StatAggregationMethod.Average);
        statsRecorder.Add($"{MetricPrefix}/EpisodeLength", summary.EpisodeLength, StatAggregationMethod.Average);
        statsRecorder.Add($"{MetricPrefix}/TotalReward", summary.TotalReward, StatAggregationMethod.Average);
    }
}
