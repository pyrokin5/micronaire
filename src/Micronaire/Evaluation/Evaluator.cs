// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
using System.Net;
using Micronaire.Claims;
using Micronaire.GenerationClaimEvaluation;
using Micronaire.GroundTruth;
using Micronaire.LLMEvaluation;
using Micronaire.OverallClaimEvaluation;
using Micronaire.RetrievalClaimEvaluation;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Http.Resilience;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Newtonsoft.Json;

namespace Micronaire;

/// <summary>
/// Evaluates RAG Pipeline responses with respect to a query.
/// </summary>
public class Evaluator : IEvaluator
{
    private readonly ILogger<Evaluator> logger;
    private readonly ILLMEvaluator llmEvaluator;
    private readonly IClaimExtractor claimExtractor;
    private readonly IConfiguration configuration;
    private readonly IOverallClaimEvaluator overallClaimEvaluator;
    private readonly IRetrievalClaimEvaluator retrievalClaimEvaluator;
    private readonly IGenerationClaimEvaluator generationClaimEvaluator;

    /// <summary>
    /// Initializes a new instance of the <see cref="Evaluator"/> class.
    /// </summary>
    /// <param name="logger">Logging interface dependency. Used for logging information, warnings, and errors during the evaluation process.</param>
    /// <param name="configuration">Configuration interface dependency. Provides access to configuration settings required for the evaluator.</param>
    /// <param name="claimExtractor">Claim extractor interface dependency. Used for extracting claims from text chunks during the evaluation process.</param>
    /// <param name="llmEvaluator">LLM evaluator interface dependency. Used for evaluating the generated answers against the ground truth using a language model.</param>
    /// <param name="overallClaimEvaluator">Overall claim evaluator interface dependency. Used for evaluating the overall metrics based on claim analysis of the generated and ground truth answers.</param>
    /// <param name="retrievalClaimEvaluator">Retrieval claim evaluator interface dependency. Used for evaluating the retrieval process using the context and ground truth claims.</param>
    /// <param name="generationClaimEvaluator">Generation claim evaluator interface dependency. Used for evaluating the generation process using the context, generated, and ground truth claims.</param>
    public Evaluator(
        ILogger<Evaluator> logger,
        IConfiguration configuration,
        IClaimExtractor claimExtractor,
        ILLMEvaluator llmEvaluator,
        IOverallClaimEvaluator overallClaimEvaluator,
        IRetrievalClaimEvaluator retrievalClaimEvaluator,
        IGenerationClaimEvaluator generationClaimEvaluator)
    {
        this.logger = logger;
        this.configuration = configuration;
        this.claimExtractor = claimExtractor;
        this.llmEvaluator = llmEvaluator;
        this.overallClaimEvaluator = overallClaimEvaluator;
        this.retrievalClaimEvaluator = retrievalClaimEvaluator;
        this.generationClaimEvaluator = generationClaimEvaluator;
    }

    /// <inheritdoc/>
    public async Task<EvaluationReport> EvaluateAsync(
        IRagPipeline pipeline,
        string groundTruthPath,
        IKernelBuilder? kernelBuilder = null,
        CancellationToken cancellationToken = default)
    {
        var questionReports = new List<QuestionReport>();
        Kernel evaluator;
        if (kernelBuilder != null)
        {
            evaluator = this.BuildLocalEvaluator(kernelBuilder);
        }
        else
        {
            evaluator = this.BuildEvaluator();
        }

        this.logger.LogInformation("Loading ground truth from {path}", groundTruthPath);
        foreach (
            var (question, groundTruthAnswer) in GroundTruthLoader.LoadQADataSet(groundTruthPath))
        {
            this.logger.LogInformation("Evaluating question: {question}", question);
            var (generatedAnswer, contexts) = await pipeline.GenerateAsync(
                question,
                cancellationToken);
            this.logger.LogInformation("Generated answer: {generatedAnswer}", generatedAnswer);

            this.logger.LogInformation("Extracting claims for question: {question}", question);
            var contextClaimsRaw = await Task.WhenAll(
                contexts
                    .Select(c => c.Context)
                    .Select(async c =>
                        (
                            await this.claimExtractor.ExtractClaimsAsync(
                                evaluator,
                                c,
                                cancellationToken)).Where(c => !c.IsTriplet)));
            var contextClaims = contextClaimsRaw.SelectMany(c => c);
            var groundTruthAnswerClaimsRaw = await this.claimExtractor.ExtractClaimsAsync(
                evaluator,
                groundTruthAnswer,
                cancellationToken);
            var groundTruthClaims = groundTruthAnswerClaimsRaw.Where(c => !c.IsTriplet);
            var generatedClaimsRaw = await this.claimExtractor.ExtractClaimsAsync(
                evaluator,
                generatedAnswer,
                cancellationToken);
            var generatedClaims = generatedClaimsRaw.Where(c => !c.IsTriplet);

            this.logger.LogInformation("Generating reports for question {question}", question);
            var llmReport = await this.llmEvaluator.EvaluateAsync(
                evaluator,
                question,
                string.Join('\n', contexts.Select(c => c.Context)),
                generatedAnswer,
                groundTruthAnswer,
                cancellationToken);
            var overallClaimReport = await this.overallClaimEvaluator.EvaluateAsync(
                evaluator,
                generatedClaims,
                groundTruthClaims,
                cancellationToken);
            var retrievalClaimReport = await this.retrievalClaimEvaluator.EvaluateAsync(
                evaluator,
                groundTruthClaims,
                contextClaims,
                cancellationToken);
            var generationClaimReport =
                await this.generationClaimEvaluator.EvaluateGeneratorMetricsAsync(
                    evaluator,
                    contextClaims,
                    contextClaimsRaw,
                    generatedClaims,
                    groundTruthClaims,
                    cancellationToken);
            questionReports.Add(
                new QuestionReport()
                {
                    Question = question,
                    LLMReport = llmReport,
                    OverallClaimReport = overallClaimReport,
                    RetrievalClaimReport = retrievalClaimReport,
                    GenerationClaimReport = generationClaimReport,
                });
        }

        var evaluationReport = this.SummarizeQuestionReports(questionReports);
        this.logger.LogInformation(
            "EvaluationReport: {evaluationReport}",
            JsonConvert.SerializeObject(evaluationReport, Formatting.Indented));
        return evaluationReport;
    }

    private EvaluationReport SummarizeQuestionReports(IEnumerable<QuestionReport> questionReports)
    {
        // average all the floats in the LLM reports
        var averageLLMReport = new LLMEvaluationReport()
        {
            Groundedness = questionReports.Average(q => q.LLMReport.Groundedness),
            Relevance = questionReports.Average(q => q.LLMReport.Relevance),
            Coherence = questionReports.Average(q => q.LLMReport.Coherence),
            Fluency = questionReports.Average(q => q.LLMReport.Fluency),
            RetrievalScore = questionReports.Average(q => q.LLMReport.RetrievalScore),
            Similarity = questionReports.Average(q => q.LLMReport.Similarity),
        };
        var averageOverallClaimReport = new OverallClaimReport()
        {
            Precision = questionReports.Average(q => q.OverallClaimReport.Precision),
            Recall = questionReports.Average(q => q.OverallClaimReport.Recall),
            F1Score = questionReports.Average(q => q.OverallClaimReport.F1Score),
        };
        var averageRetrievalClaimReport = new RetrievalClaimReport()
        {
            ClaimRecall = questionReports.Average(q => q.RetrievalClaimReport.ClaimRecall),
            ContextPrecision = questionReports.Average(q =>
                q.RetrievalClaimReport.ContextPrecision),
        };
        var averageGenerationClaimReport = new GenerationClaimReport()
        {
            Faithfulness = questionReports.Average(q => q.GenerationClaimReport.Faithfulness),
            RelevantNoiseSensitivity = questionReports.Average(q =>
                q.GenerationClaimReport.RelevantNoiseSensitivity),
            IrrelevantNoiseSensitivity = questionReports.Average(q =>
                q.GenerationClaimReport.IrrelevantNoiseSensitivity),
            Hallucination = questionReports.Average(q => q.GenerationClaimReport.Hallucination),
            SelfKnowledgeScore = questionReports.Average(q =>
                q.GenerationClaimReport.SelfKnowledgeScore),
            ContextUtilization = questionReports.Average(q =>
                q.GenerationClaimReport.ContextUtilization),
        };

        return new()
        {
            QuestionReports = questionReports,
            AverageLLMReport = averageLLMReport,
            AverageOverallClaimReport = averageOverallClaimReport,
            AverageRetrievalClaimReport = averageRetrievalClaimReport,
            AverageGenerationClaimReport = averageGenerationClaimReport,
        };
    }

    private Kernel BuildEvaluator()
    {
        var evaluatorKernelConfig =
            this.configuration
                .GetRequiredSection(nameof(EvaluatorAzureOpenAIConfig))
                .Get<EvaluatorAzureOpenAIConfig>()
            ?? throw new NullReferenceException($"Missing {nameof(EvaluatorAzureOpenAIConfig)}");
        var evaluatorKernelBuilder = Kernel
            .CreateBuilder()
            .AddAzureOpenAIChatCompletion(
                evaluatorKernelConfig.ChatModelDeploymentName,
                evaluatorKernelConfig.Endpoint,
                evaluatorKernelConfig.ApiKey);
        evaluatorKernelBuilder.Services.ConfigureHttpClientDefaults(c =>
        {
            // Use a standard resiliency policy, augmented to retry on 401 Unauthorized for this example
            c.AddStandardResilienceHandler()
                .Configure(o =>
                {
                    o.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(2);
                    o.Retry.ShouldHandle = args =>
                        ValueTask.FromResult(
                            args.Outcome.Result?.StatusCode is HttpStatusCode.Unauthorized
                                || args.Outcome.Result?.StatusCode is HttpStatusCode.TooManyRequests
                                || args.Outcome.Exception?.InnerException is TaskCanceledException);
                    o.Retry.MaxRetryAttempts = 2;
                    o.Retry.BackoffType = Polly.DelayBackoffType.Constant;
                    o.Retry.UseJitter = false;
                    o.Retry.Delay = TimeSpan.FromSeconds(40.0);
                });
        });
        var kernel = evaluatorKernelBuilder.Build();

        // add claim extraction plugin to the kernel
        kernel.ImportPluginFromPromptDirectory(
            Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "..\\..\\..\\..\\Micronaire\\Claims\\ExtractorPlugins"));

        // add evaluation plugins to the kernel
        kernel.ImportPluginFromPromptDirectory(
            Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "..\\..\\..\\..\\Micronaire\\LLMEvaluation\\Plugins"));
        return kernel;
    }

    private Kernel BuildLocalEvaluator(IKernelBuilder evaluatorKernelBuilder)
    {
        evaluatorKernelBuilder.Services.ConfigureHttpClientDefaults(c =>
        {
            // Use a standard resiliency policy, augmented to retry on 401 Unauthorized for this example
            c.AddStandardResilienceHandler()
                .Configure(o =>
                {
                    o.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(2);
                    o.Retry.ShouldHandle = args =>
                        ValueTask.FromResult(
                            args.Outcome.Result?.StatusCode is HttpStatusCode.Unauthorized
                                || args.Outcome.Result?.StatusCode is HttpStatusCode.TooManyRequests
                                || args.Outcome.Exception?.InnerException is TaskCanceledException);
                    o.Retry.MaxRetryAttempts = 2;
                    o.Retry.BackoffType = Polly.DelayBackoffType.Constant;
                    o.Retry.UseJitter = false;
                    o.Retry.Delay = TimeSpan.FromSeconds(40.0);
                });
        });
        var kernel = evaluatorKernelBuilder.Build();

        // add claim extraction plugin to the kernel
        kernel.ImportPluginFromPromptDirectory(
            Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "..\\..\\..\\..\\Micronaire\\Claims\\ExtractorPlugins"));

        // add evaluation plugins to the kernel
        kernel.ImportPluginFromPromptDirectory(
            Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "..\\..\\..\\..\\Micronaire\\LLMEvaluation\\Plugins"));

        return kernel;
    }
}
