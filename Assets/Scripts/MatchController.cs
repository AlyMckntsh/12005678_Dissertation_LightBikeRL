using System;
using System.Collections;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif
using Unity.InferenceEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;

/// <summary>
/// Runs the fixed dissertation experiment: PPO learner, heuristic opponent,
/// parallel arenas, two reward conditions, and a small evaluation mode.
/// It is also the run-level coordinator that keeps both bikes requesting decisions on the same shared step.
/// </summary>
[DefaultExecutionOrder(-150)]
public class MatchController : MonoBehaviour
{
    /// <summary>
    /// Selects whether the scene is collecting training data or running a fixed evaluation pass.
    /// </summary>
    public enum RunMode
    {
        Training = 0,
        Evaluation = 1
    }

    /// <summary>
    /// Holds the scene references for one learner-versus-opponent arena pair.
    /// </summary>
    [Serializable]
    public struct ArenaSlot
    {
        [SerializeField] private ArenaContext arenaContext;
        [SerializeField] private PlayerBikeController learnerBike;
        [SerializeField] private PlayerBikeController opponentBike;

        public ArenaContext ArenaContext
        {
            get { return arenaContext; }
        }

        public PlayerBikeController LearnerBike
        {
            get { return learnerBike; }
        }

        public PlayerBikeController OpponentBike
        {
            get { return opponentBike; }
        }
    }

    /// <summary>
    /// Adds the runtime-only state that MatchController builds on top of the serialized arena slot.
    /// </summary>
    private sealed class ArenaRuntime
    {
        internal ArenaSlot Slot;
        internal LightBikeAgent LearnerAgent;
        internal LightBikeAgent OpponentAgent;
    }

    // This must match the behaviour key in the training YAML and the behaviour name used by the agents.
    private const string BehaviorName = "lightbike_ppo";
    // Team ids keep the learner and heuristic opponent logically separated inside ML-Agents.
    private const int LearnerTeamId = 0;
    private const int OpponentTeamId = 1;

    [Header("Arena Wiring")]
    [SerializeField] private List<ArenaSlot> arenas = new List<ArenaSlot>();

    [Header("Run Mode")]
    [SerializeField] private RunMode runMode = RunMode.Training;
    [SerializeField] private ModelAsset evaluationModel;

    [Header("Reward Condition")]
    [SerializeField] private RewardManager.RewardCondition rewardCondition = RewardManager.RewardCondition.SurvivalOnly;

    [Header("Run Setup")]
    [SerializeField] private int baseSeed = 1;

    [Header("Evaluation")]
    [SerializeField, Tooltip("Total learner episodes to evaluate across all arenas combined."), Min(1)]
    private int evaluationEpisodeCount = 100;

    // These lookup tables let the controller jump from bikes and agents back to the arena they belong to.
    private readonly List<ArenaRuntime> _arenaRuntimes = new List<ArenaRuntime>();
    private readonly Dictionary<PlayerBikeController, ArenaRuntime> _arenaByLearnerBike =
        new Dictionary<PlayerBikeController, ArenaRuntime>();
    private readonly Dictionary<LightBikeAgent, ArenaRuntime> _arenaByLearnerAgent =
        new Dictionary<LightBikeAgent, ArenaRuntime>();
    private readonly TrainingStatsReporter _trainingStatsReporter = new TrainingStatsReporter();
    private readonly EvaluationCsvExporter _evaluationCsvExporter = new EvaluationCsvExporter();

    private bool _isConfigured;
    private int _completedEvaluationEpisodes;
    private int _evaluationWins;
    private int _evaluationDraws;
    private float _evaluationRewardTotal;
    private int _evaluationEpisodeLengthTotal;

    private void Start()
    {
        // Fail early on broken scene wiring before any training or evaluation starts.
        if (!ValidateAndBuildArenaRuntimes())
        {
            return;
        }

        if (runMode == RunMode.Evaluation && evaluationModel == null)
        {
            StopRun("Evaluation mode requires an evaluation model.");
            return;
        }

        // Use one fixed seed for both gameplay randomness and ML-Agents inference so runs are repeatable.
        UnityEngine.Random.InitState(baseSeed);
        Academy.Instance.InferenceSeed = baseSeed;
        ResetEvaluationTotals();

        // Once every arena is configured, the controller can start driving decisions itself.
        if (!ConfigureAllArenas())
        {
            return;
        }

        _isConfigured = true;
        SubscribeToArenaEvents();
        StartCoroutine(RequestInitialDecisionsAfterStartup());
        Debug.Log(
            $"[MatchController] mode={runMode}, arenas={_arenaRuntimes.Count}, rewardCondition={rewardCondition}, seed={baseSeed}",
            this);
    }

    private void OnDestroy()
    {
        // Remove event hooks when the scene stops so a domain reload cannot leave stale subscriptions behind.
        UnsubscribeAllArenaEvents();
    }

    private bool ValidateAndBuildArenaRuntimes()
    {
        // Flatten the serialized slots into runtime records and fast lookup tables.
        _arenaRuntimes.Clear();
        _arenaByLearnerBike.Clear();
        _arenaByLearnerAgent.Clear();

        if (arenas == null || arenas.Count == 0)
        {
            StopRun("At least one arena slot is required.");
            return false;
        }

        HashSet<PlayerBikeController> learnerBikeSet = new HashSet<PlayerBikeController>();

        for (int i = 0; i < arenas.Count; i++)
        {
            ArenaSlot slot = arenas[i];
            if (!ValidateArenaSlot(slot, i, learnerBikeSet))
            {
                return false;
            }

            ArenaRuntime runtime = new ArenaRuntime
            {
                Slot = slot
            };

            _arenaRuntimes.Add(runtime);
            _arenaByLearnerBike[slot.LearnerBike] = runtime;
        }

        return true;
    }

    private bool ValidateArenaSlot(
        ArenaSlot slot,
        int slotIndex,
        HashSet<PlayerBikeController> learnerBikeSet)
    {
        // Check the scene wiring here so training never starts with mismatched arena state.
        if (slot.ArenaContext == null || slot.LearnerBike == null || slot.OpponentBike == null)
        {
            StopRun($"Arena slot {slotIndex} is missing ArenaContext, learner bike, or opponent bike.");
            return false;
        }

        if (slot.LearnerBike == slot.OpponentBike)
        {
            StopRun($"Arena slot {slotIndex} uses the same bike for learner and opponent.");
            return false;
        }

        if (slot.LearnerBike.ArenaContext != slot.ArenaContext || slot.OpponentBike.ArenaContext != slot.ArenaContext)
        {
            StopRun($"Arena slot {slotIndex} bikes must both reference the same ArenaContext as the slot.");
            return false;
        }

        if (!learnerBikeSet.Add(slot.LearnerBike))
        {
            StopRun($"Learner bike {slot.LearnerBike.name} is assigned to multiple arena slots.");
            return false;
        }

        Vector2Int minCell = slot.ArenaContext.MinCell;
        Vector2Int maxCell = slot.ArenaContext.MaxCell;

        if (!IsWithinBounds(slot.LearnerBike.StartCell, minCell, maxCell))
        {
            StopRun($"Arena slot {slotIndex} learner start cell {slot.LearnerBike.StartCell} is outside arena bounds.");
            return false;
        }

        if (!IsWithinBounds(slot.OpponentBike.StartCell, minCell, maxCell))
        {
            StopRun($"Arena slot {slotIndex} opponent start cell {slot.OpponentBike.StartCell} is outside arena bounds.");
            return false;
        }

        if (slot.LearnerBike.StartCell == slot.OpponentBike.StartCell)
        {
            StopRun($"Arena slot {slotIndex} learner and opponent share the same start cell.");
            return false;
        }

        if (slot.LearnerBike.EffectiveStartDirection + slot.OpponentBike.EffectiveStartDirection != Vector2Int.zero)
        {
            // The dissertation setup expects the bikes to face each other at the start of every round.
            StopRun($"Arena slot {slotIndex} learner and opponent must start facing opposite directions.");
            return false;
        }

        if (!Mathf.Approximately(slot.LearnerBike.StepDuration, slot.OpponentBike.StepDuration))
        {
            // Shared-step simulation assumes both bikes advance on the same clock.
            StopRun(
                $"Arena slot {slotIndex} has a stepDuration mismatch " +
                $"(learner={slot.LearnerBike.StepDuration:F4}, opponent={slot.OpponentBike.StepDuration:F4}).");
            return false;
        }

        return true;
    }

    private static bool IsWithinBounds(Vector2Int cell, Vector2Int minCell, Vector2Int maxCell)
    {
        // Use the ArenaContext bounds already normalised by MinCell and MaxCell.
        return cell.x >= minCell.x &&
               cell.x <= maxCell.x &&
               cell.y >= minCell.y &&
               cell.y <= maxCell.y;
    }

    private bool ConfigureAllArenas()
    {
        for (int i = 0; i < _arenaRuntimes.Count; i++)
        {
            ArenaRuntime runtime = _arenaRuntimes[i];

            // Each runtime ends up with two active agents, but the learner is the only one we report externally.
            if (!ConfigureLearnerBike(runtime) || !ConfigureOpponentBike(runtime))
            {
                return false;
            }

            _arenaByLearnerAgent[runtime.LearnerAgent] = runtime;
            runtime.LearnerAgent.EpisodeCompleted += HandleLearnerEpisodeCompleted;

            // Keep both bikes on the same fixed reward condition for a round.
            // Only the learner feeds TensorBoard, but both active agents still complete episodes.
            ApplyRewardCondition(runtime.Slot.LearnerBike);
            ApplyRewardCondition(runtime.Slot.OpponentBike);
        }

        return true;
    }

    private bool ConfigureLearnerBike(ArenaRuntime runtime)
    {
        // The learner trains in Default mode, or runs a frozen ONNX model during evaluation.
        BehaviorType behaviorType = runMode == RunMode.Evaluation
            ? BehaviorType.InferenceOnly
            : BehaviorType.Default;
        ModelAsset model = runMode == RunMode.Evaluation ? evaluationModel : null;

        return ConfigureBike(runtime.Slot.LearnerBike, LearnerTeamId, behaviorType, model, out runtime.LearnerAgent);
    }

    private bool ConfigureOpponentBike(ArenaRuntime runtime)
    {
        // The opponent always uses the built-in heuristic policy as the fixed baseline.
        return ConfigureBike(runtime.Slot.OpponentBike, OpponentTeamId, BehaviorType.HeuristicOnly, null, out runtime.OpponentAgent);
    }

    private bool ConfigureBike(
        PlayerBikeController bike,
        int teamId,
        BehaviorType behaviorType,
        ModelAsset model,
        out LightBikeAgent agent)
    {
        agent = null;

        if (bike == null)
        {
            StopRun("Encountered a null bike while configuring an arena.");
            return false;
        }

        agent = bike.GetComponent<LightBikeAgent>();
        BehaviorParameters behaviorParameters = bike.GetComponent<BehaviorParameters>();
        DecisionRequester decisionRequester = bike.GetComponent<DecisionRequester>();
        RewardManager rewardManager = bike.GetComponent<RewardManager>();

        if (agent == null || behaviorParameters == null || decisionRequester == null || rewardManager == null)
        {
            StopRun($"Bike {bike.name} is missing LightBikeAgent, BehaviorParameters, DecisionRequester, or RewardManager.");
            return false;
        }

        // Keep both bikes under the same behaviour name so the learner and heuristic share one action interface.
        behaviorParameters.BehaviorName = BehaviorName;
        behaviorParameters.TeamId = teamId;
        behaviorParameters.Model = model;
        behaviorParameters.BehaviorType = behaviorType;

        // DecisionRequester stays for component wiring, but MatchController requests decisions manually
        // so both bikes choose their next move on the same synchronized step.
        decisionRequester.DecisionPeriod = 1;
        decisionRequester.DecisionStep = 0;
        decisionRequester.TakeActionsBetweenDecisions = true;
        decisionRequester.enabled = false;

        // LightBikeAgent handles episode flow. MatchController only decides how this bike should behave for the run.
        agent.enabled = true;
        return true;
    }

    private void ApplyRewardCondition(PlayerBikeController bike)
    {
        RewardManager rewardManager = bike.GetComponent<RewardManager>();
        if (rewardManager != null)
        {
            // Apply the same reward formula to both bikes so each round uses one consistent scoring rule.
            rewardManager.SetRewardCondition(rewardCondition);
        }
    }

    private void SubscribeToArenaEvents()
    {
        // Learner steps and arena resets are the two moments when the next decision pair should be requested.
        for (int i = 0; i < _arenaRuntimes.Count; i++)
        {
            ArenaRuntime runtime = _arenaRuntimes[i];
            runtime.Slot.LearnerBike.StepAdvanced += HandleLearnerStepAdvanced;
            runtime.Slot.ArenaContext.RoundResetCompleted += HandleArenaRoundResetCompleted;
        }
    }

    private void UnsubscribeAllArenaEvents()
    {
        // Mirror every subscription made during setup so reloading the scene cannot duplicate callbacks.
        for (int i = 0; i < _arenaRuntimes.Count; i++)
        {
            ArenaRuntime runtime = _arenaRuntimes[i];

            if (runtime.Slot.LearnerBike != null)
            {
                runtime.Slot.LearnerBike.StepAdvanced -= HandleLearnerStepAdvanced;
            }

            if (runtime.Slot.ArenaContext != null)
            {
                runtime.Slot.ArenaContext.RoundResetCompleted -= HandleArenaRoundResetCompleted;
            }

            if (runtime.LearnerAgent != null)
            {
                runtime.LearnerAgent.EpisodeCompleted -= HandleLearnerEpisodeCompleted;
            }
        }

        _arenaByLearnerAgent.Clear();
    }

    private void HandleLearnerStepAdvanced(PlayerBikeController learnerBike)
    {
        if (!_isConfigured || learnerBike == null)
        {
            return;
        }

        if (!_arenaByLearnerBike.TryGetValue(learnerBike, out ArenaRuntime runtime))
        {
            return;
        }

        // As soon as the shared step finishes, queue the next learner and opponent decisions together.
        RequestDecisionsForArena(runtime);
    }

    private IEnumerator RequestInitialDecisionsAfterStartup()
    {
        // Wait until the bikes have finished their Start() reset before asking either agent
        // for the first action of the run.
        yield return null;

        if (!_isConfigured)
        {
            yield break;
        }

        for (int i = 0; i < _arenaRuntimes.Count; i++)
        {
            RequestDecisionsForArena(_arenaRuntimes[i]);
        }
    }

    private void HandleArenaRoundResetCompleted(ArenaContext arenaContext)
    {
        if (!_isConfigured || arenaContext == null)
        {
            return;
        }

        // After a reset, request the opening move pair for that arena only.
        for (int i = 0; i < _arenaRuntimes.Count; i++)
        {
            ArenaRuntime runtime = _arenaRuntimes[i];
            if (runtime.Slot.ArenaContext == arenaContext)
            {
                RequestDecisionsForArena(runtime);
                return;
            }
        }
    }

    private static void RequestDecisionsForArena(ArenaRuntime runtime)
    {
        if (runtime == null)
        {
            return;
        }

        // Request both decisions back-to-back so the upcoming shared step uses one coherent pair of actions.
        RequestDecision(runtime.LearnerAgent, runtime.Slot.LearnerBike);
        RequestDecision(runtime.OpponentAgent, runtime.Slot.OpponentBike);
    }

    private static void RequestDecision(LightBikeAgent agent, PlayerBikeController bike)
    {
        if (agent == null || !agent.isActiveAndEnabled || bike == null || bike.IsDead)
        {
            return;
        }

        // DecisionRequester is disabled, so this is the explicit trigger for one fresh action.
        agent.RequestDecision();
    }

    private void HandleLearnerEpisodeCompleted(LightBikeAgent agent, LightBikeAgent.EpisodeSummary summary)
    {
        if (!_isConfigured || agent == null || !_arenaByLearnerAgent.ContainsKey(agent))
        {
            return;
        }

        // Only learner summaries feed the recorded metrics. Opponent episodes still complete, but stay internal.
        if (runMode == RunMode.Training)
        {
            _trainingStatsReporter.RecordEpisode(summary);
            return;
        }

        RecordEvaluationEpisode(summary);
    }

    private void RecordEvaluationEpisode(LightBikeAgent.EpisodeSummary summary)
    {
        // Collect aggregate evaluation statistics until the configured episode quota is reached.
        _completedEvaluationEpisodes++;

        if (summary.DidWin)
        {
            _evaluationWins++;
        }
        else if (summary.DidDraw)
        {
            _evaluationDraws++;
        }

        _evaluationRewardTotal += summary.TotalReward;
        _evaluationEpisodeLengthTotal += summary.EpisodeLength;

        if (_completedEvaluationEpisodes < evaluationEpisodeCount)
        {
            return;
        }

        float winRate = _completedEvaluationEpisodes > 0
            ? _evaluationWins / (float)_completedEvaluationEpisodes
            : 0f;
        float averageReward = _completedEvaluationEpisodes > 0
            ? _evaluationRewardTotal / _completedEvaluationEpisodes
            : 0f;
        float averageEpisodeLength = _completedEvaluationEpisodes > 0
            ? _evaluationEpisodeLengthTotal / (float)_completedEvaluationEpisodes
            : 0f;
        int losses = Mathf.Max(0, _completedEvaluationEpisodes - _evaluationWins - _evaluationDraws);
        EvaluationRunSummary runSummary = new EvaluationRunSummary(
            baseSeed,
            rewardCondition,
            GetEvaluationModelDescriptor(),
            _completedEvaluationEpisodes,
            _evaluationWins,
            _evaluationDraws,
            losses,
            winRate,
            averageReward,
            averageEpisodeLength);
        bool exportedToCsv = _evaluationCsvExporter.TryAppendRunSummary(runSummary, out string csvPath);

        Debug.Log(
            $"[MatchController] Evaluation summary: episodes={_completedEvaluationEpisodes}, wins={_evaluationWins}, draws={_evaluationDraws}, losses={losses}, " +
            $"winRate={winRate:F3}, averageReward={averageReward:F3}, averageEpisodeLength={averageEpisodeLength:F1}" +
            (exportedToCsv ? $", csvPath={csvPath}" : string.Empty),
            this);

        // End the run immediately once the requested evaluation sample has been collected.
        StopEvaluationRun();
    }

    private string GetEvaluationModelDescriptor()
    {
        if (evaluationModel == null)
        {
            return string.Empty;
        }

#if UNITY_EDITOR
        string assetPath = AssetDatabase.GetAssetPath(evaluationModel);
        if (!string.IsNullOrEmpty(assetPath))
        {
            return assetPath;
        }
#endif

        return evaluationModel.name;
    }

    private void ResetEvaluationTotals()
    {
        // Start each evaluation run with a clean accumulator.
        _completedEvaluationEpisodes = 0;
        _evaluationWins = 0;
        _evaluationDraws = 0;
        _evaluationRewardTotal = 0f;
        _evaluationEpisodeLengthTotal = 0;
    }

    private void StopRun(string reason)
    {
        // Training just disables the controller, while evaluation also exits so automation can detect the failure.
        Debug.LogError($"[MatchController] {reason}", this);
        _isConfigured = false;
        enabled = false;

        if (runMode == RunMode.Evaluation)
        {
            StopEvaluationRun(1);
        }
    }

    private void StopEvaluationRun(int exitCode = 0)
    {
        _isConfigured = false;
        enabled = false;

        // End play mode in the editor, or return an exit code when running a built player.
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit(exitCode);
#endif
    }
}
