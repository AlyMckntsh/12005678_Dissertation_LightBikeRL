using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// Appends one CSV row for each completed evaluation run so fixed-policy tests leave a persistent audit trail.
/// </summary>
internal sealed class EvaluationCsvExporter
{
    private const string EvaluationCsvFileName = "evaluation_runs.csv";
    private const string CsvHeader =
        "seed,reward_condition,model_checkpoint,episodes,wins,draws,losses,win_rate,avg_reward,avg_episode_length";

    internal bool TryAppendRunSummary(EvaluationRunSummary runSummary, out string csvPath)
    {
        csvPath = GetCsvPath();

        try
        {
            string directoryPath = Path.GetDirectoryName(csvPath);
            if (!string.IsNullOrEmpty(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }

            bool writeHeader = !File.Exists(csvPath) || new FileInfo(csvPath).Length == 0;
            using (StreamWriter writer = new StreamWriter(csvPath, true, new UTF8Encoding(false)))
            {
                if (writeHeader)
                {
                    writer.WriteLine(CsvHeader);
                }

                writer.WriteLine(BuildCsvRow(runSummary));
            }

            return true;
        }
        catch (Exception exception)
        {
            Debug.LogError($"[EvaluationCsvExporter] Failed to append evaluation CSV '{csvPath}': {exception.Message}");
            return false;
        }
    }

    private static string GetCsvPath()
    {
        string projectRootPath = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        return Path.Combine(projectRootPath, "results", EvaluationCsvFileName);
    }

    private static string BuildCsvRow(EvaluationRunSummary runSummary)
    {
        StringBuilder builder = new StringBuilder(160);
        AppendCsvField(builder, runSummary.Seed.ToString(CultureInfo.InvariantCulture));
        AppendCsvField(builder, runSummary.RewardCondition.ToString());
        AppendCsvField(builder, runSummary.ModelCheckpoint);
        AppendCsvField(builder, runSummary.Episodes.ToString(CultureInfo.InvariantCulture));
        AppendCsvField(builder, runSummary.Wins.ToString(CultureInfo.InvariantCulture));
        AppendCsvField(builder, runSummary.Draws.ToString(CultureInfo.InvariantCulture));
        AppendCsvField(builder, runSummary.Losses.ToString(CultureInfo.InvariantCulture));
        AppendCsvField(builder, runSummary.WinRate.ToString("F6", CultureInfo.InvariantCulture));
        AppendCsvField(builder, runSummary.AverageReward.ToString("F6", CultureInfo.InvariantCulture));
        AppendCsvField(builder, runSummary.AverageEpisodeLength.ToString("F6", CultureInfo.InvariantCulture));
        return builder.ToString();
    }

    private static void AppendCsvField(StringBuilder builder, string value)
    {
        if (builder.Length > 0)
        {
            builder.Append(',');
        }

        builder.Append(EscapeCsv(value));
    }

    private static string EscapeCsv(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        bool requiresQuotes =
            value.IndexOf(',') >= 0 ||
            value.IndexOf('"') >= 0 ||
            value.IndexOf('\n') >= 0 ||
            value.IndexOf('\r') >= 0;

        if (!requiresQuotes)
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }
}

/// <summary>
/// Stores the aggregated metrics for one completed evaluation run.
/// </summary>
internal readonly struct EvaluationRunSummary
{
    internal EvaluationRunSummary(
        int seed,
        RewardManager.RewardCondition rewardCondition,
        string modelCheckpoint,
        int episodes,
        int wins,
        int draws,
        int losses,
        float winRate,
        float averageReward,
        float averageEpisodeLength)
    {
        Seed = seed;
        RewardCondition = rewardCondition;
        ModelCheckpoint = modelCheckpoint ?? string.Empty;
        Episodes = episodes;
        Wins = wins;
        Draws = draws;
        Losses = losses;
        WinRate = winRate;
        AverageReward = averageReward;
        AverageEpisodeLength = averageEpisodeLength;
    }

    internal readonly int Seed;
    internal readonly RewardManager.RewardCondition RewardCondition;
    internal readonly string ModelCheckpoint;
    internal readonly int Episodes;
    internal readonly int Wins;
    internal readonly int Draws;
    internal readonly int Losses;
    internal readonly float WinRate;
    internal readonly float AverageReward;
    internal readonly float AverageEpisodeLength;
}
