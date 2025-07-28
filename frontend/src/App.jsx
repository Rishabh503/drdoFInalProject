import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, 
  Play, 
  Download, 
  FileText, 
  BarChart3, 
  Brain, 
  Target, 
  TrendingUp, 
  AlertCircle,
  CheckCircle,
  Loader,
  Clock,
  Database,
  Activity,
  Zap,
  Settings,
  Info,
  Cpu,
  Layers,
  GitBranch
} from 'lucide-react';

const App = () => {
  const [cipherFile, setCipherFile] = useState(null);
  const [randomFile, setRandomFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('random_forest');
  const [availableModels, setAvailableModels] = useState({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const cipherInputRef = useRef(null);
  const randomInputRef = useRef(null);

  // Fetch available models on component mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models');
      const models = await response.json();
      setAvailableModels(models);
    } catch (err) {
      console.error('Failed to fetch available models:', err);
      // Fallback models if API fails
      setAvailableModels({
        'random_forest': {
          name: 'Random Forest',
          description: 'Ensemble method using multiple decision trees',
          strengths: ['Fast training', 'Good interpretability', 'Handles overfitting well']
        },
        'xgboost': {
          name: 'XGBoost',
          description: 'Gradient boosting framework optimized for speed and performance',
          strengths: ['High accuracy', 'Fast SHAP computation', 'Advanced regularization']
        },
        'gradient_boosting': {
          name: 'Gradient Boosting',
          description: 'Sequential ensemble method building models iteratively',
          strengths: ['Good performance', 'Built-in feature selection', 'Robust to outliers']
        }
      });
    }
  };

  const handleFileUpload = (file, type) => {
    if (file && file.type === 'text/csv') {
      if (type === 'cipher') {
        setCipherFile(file);
      } else {
        setRandomFile(file);
      }
      setError(null);
    } else {
      setError('Please upload valid CSV files only');
    }
  };

  const analyzeData = async () => {
    if (!cipherFile || !randomFile) {
      setError('Please upload both cipher and random CSV files');
      return;
    }

    if (!selectedModel) {
      setError('Please select a model for analysis');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('cipher_file', cipherFile);
    formData.append('random_file', randomFile);
    formData.append('model_type', selectedModel);

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResults(data);
      } else {
        setError(data.error || 'Analysis failed');
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure the Flask backend is running.');
      console.error('Error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const reset = () => {
    setCipherFile(null);
    setRandomFile(null);
    setResults(null);
    setError(null);
    if (cipherInputRef.current) cipherInputRef.current.value = '';
    if (randomInputRef.current) randomInputRef.current.value = '';
  };

  const formatNumber = (num, decimals = 4) => {
    return typeof num === 'number' ? num.toFixed(decimals) : num;
  };

  const getModelIcon = (modelKey) => {
    switch (modelKey) {
      case 'random_forest':
        return <GitBranch className="h-5 w-5" />;
      case 'xgboost':
        return <Zap className="h-5 w-5" />;
      case 'gradient_boosting':
        return <Layers className="h-5 w-5" />;
      default:
        return <Cpu className="h-5 w-5" />;
    }
  };

  const getModelColor = (modelKey) => {
    switch (modelKey) {
      case 'random_forest':
        return 'from-green-600 to-emerald-600';
      case 'xgboost':
        return 'from-purple-600 to-indigo-600';
      case 'gradient_boosting':
        return 'from-orange-600 to-red-600';
      default:
        return 'from-blue-600 to-cyan-600';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                  <Brain className="h-8 w-8 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-gray-900">Multi-Model SHAP Explainable AI</h1>
                  <p className="text-gray-600">Cipher vs Random Stream Analysis with Multiple ML Models</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 text-sm text-gray-500">
                  <Activity className="h-4 w-4" />
                  <span>Machine Learning Explainability</span>
                </div>
                <div className="flex items-center space-x-2 text-sm text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                  <Settings className="h-4 w-4" />
                  <span>{Object.keys(availableModels).length} Models Available</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Model Selection Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Settings className="h-6 w-6 mr-3 text-purple-600" />
            Select Machine Learning Model
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(availableModels).map(([key, model]) => (
              <div
                key={key}
                className={`border-2 rounded-lg p-6 cursor-pointer transition-all duration-300 ${
                  selectedModel === key
                    ? 'border-blue-500 bg-blue-50 shadow-lg'
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                }`}
                onClick={() => setSelectedModel(key)}
              >
                <div className="flex items-center space-x-3 mb-3">
                  <div className={`p-2 bg-gradient-to-r ${getModelColor(key)} rounded-lg`}>
                    {getModelIcon(key)}
                    {/* <span className="text-white text-xs ml-1">{getModelIcon(key).props.className = "h-5 w-5 text-white"}</span> */}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
                    {selectedModel === key && (
                      <span className="text-sm text-blue-600 font-medium">Selected</span>
                    )}
                  </div>
                </div>
                
                <p className="text-gray-600 text-sm mb-3">{model.description}</p>
                
                <div className="space-y-1">
                  <h4 className="text-xs font-medium text-gray-700 uppercase tracking-wide">Strengths:</h4>
                  {model.strengths?.map((strength, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <CheckCircle className="h-3 w-3 text-green-500 flex-shrink-0" />
                      <span className="text-xs text-gray-600">{strength}</span>
                    </div>
                  ))}
                </div>
                
                <div className="mt-4">
                  <div className={`w-full h-2 bg-gray-200 rounded-full overflow-hidden ${
                    selectedModel === key ? 'animate-pulse' : ''
                  }`}>
                    <div className={`h-full bg-gradient-to-r ${getModelColor(key)} ${
                      selectedModel === key ? 'w-full' : 'w-0'
                    } transition-all duration-500`}></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {selectedModel && (
            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
              <div className="flex items-center space-x-2">
                <Info className="h-5 w-5 text-blue-600" />
                <span className="text-blue-900 font-medium">
                  Selected: {availableModels[selectedModel]?.name}
                </span>
              </div>
              <p className="text-blue-800 text-sm mt-1">
                {availableModels[selectedModel]?.description}
              </p>
            </div>
          )}
        </div>

        {/* File Upload Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Upload className="h-6 w-6 mr-3 text-blue-600" />
            Upload Data Files
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Cipher File Upload */}
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-blue-400 transition-colors">
              <div className="text-center">
                <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Cipher Streams CSV</h3>
                <p className="text-gray-500 mb-4">Upload your cipher data file</p>
                <input
                  ref={cipherInputRef}
                  type="file"
                  accept=".csv"
                  onChange={(e) => handleFileUpload(e.target.files[0], 'cipher')}
                  className="hidden"
                  id="cipher-upload"
                />
                <label
                  htmlFor="cipher-upload"
                  className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Choose File
                </label>
                {cipherFile && (
                  <div className="mt-3 flex items-center justify-center text-green-600">
                    <CheckCircle className="h-4 w-4 mr-2" />
                    <span className="text-sm">{cipherFile.name}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Random File Upload */}
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-purple-400 transition-colors">
              <div className="text-center">
                <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Random Streams CSV</h3>
                <p className="text-gray-500 mb-4">Upload your random data file</p>
                <input
                  ref={randomInputRef}
                  type="file"
                  accept=".csv"
                  onChange={(e) => handleFileUpload(e.target.files[0], 'random')}
                  className="hidden"
                  id="random-upload"
                />
                <label
                  htmlFor="random-upload"
                  className="inline-flex items-center px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 cursor-pointer transition-colors"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Choose File
                </label>
                {randomFile && (
                  <div className="mt-3 flex items-center justify-center text-green-600">
                    <CheckCircle className="h-4 w-4 mr-2" />
                    <span className="text-sm">{randomFile.name}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-center space-x-4">
            <button
              onClick={analyzeData}
              disabled={!cipherFile || !randomFile || !selectedModel || isAnalyzing}
              className={`inline-flex items-center px-6 py-3 rounded-lg text-white font-medium transition-all ${
                !cipherFile || !randomFile || !selectedModel || isAnalyzing
                  ? 'bg-gray-400 cursor-not-allowed'
                  : `bg-gradient-to-r ${getModelColor(selectedModel)} hover:shadow-xl transform hover:scale-105`
              }`}
            >
              {isAnalyzing ? (
                <Loader className="h-5 w-5 mr-2 animate-spin" />
              ) : (
                <Play className="h-5 w-5 mr-2" />
              )}
              {isAnalyzing ? `Analyzing with ${availableModels[selectedModel]?.name}...` : `Analyze with ${availableModels[selectedModel]?.name || 'Selected Model'}`}
            </button>
            
            <button
              onClick={reset}
              className="inline-flex items-center px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-8">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-400 mr-3" />
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isAnalyzing && (
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-8 mb-8">
            <div className="text-center">
              <div className="flex items-center justify-center mb-4">
                <div className={`p-3 bg-gradient-to-r ${getModelColor(selectedModel)} rounded-full mr-4`}>
                  {getModelIcon(selectedModel)}
                </div>
                <Loader className="h-12 w-12 text-blue-600 animate-spin" />
              </div>
              <h3 className="text-lg font-semibold text-blue-900 mb-2">
                Analyzing with {availableModels[selectedModel]?.name}...
              </h3>
              <p className="text-blue-700 mb-4">Training model and generating SHAP explanations</p>
              <div className="flex justify-center items-center space-x-6 text-sm text-blue-600">
                <div className="flex items-center">
                  <Database className="h-4 w-4 mr-1" />
                  Processing Data
                </div>
                <div className="flex items-center">
                  <Brain className="h-4 w-4 mr-1" />
                  Training {availableModels[selectedModel]?.name}
                </div>
                <div className="flex items-center">
                  <BarChart3 className="h-4 w-4 mr-1" />
                  Generating Plots
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Display */}
        {results && (
          <div className="space-y-8">
            {/* Summary Statistics */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                <Target className="h-6 w-6 mr-3 text-green-600" />
                {results.model_name} Analysis Summary
              </h2>
              
              <div className="mb-6 p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg border border-gray-200">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 bg-gradient-to-r ${getModelColor(selectedModel)} rounded-lg`}>
                    {getModelIcon(selectedModel)}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">Model: {results.model_name}</h3>
                    <p className="text-gray-600 text-sm">{availableModels[selectedModel]?.description}</p>
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <div className="flex items-center">
                    <Target className="h-8 w-8 text-green-600 mr-3" />
                    <div>
                      <p className="text-sm text-green-600 font-medium">Model Accuracy</p>
                      <p className="text-2xl font-bold text-green-900">{formatNumber(results.accuracy)}</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <div className="flex items-center">
                    <Database className="h-8 w-8 text-blue-600 mr-3" />
                    <div>
                      <p className="text-sm text-blue-600 font-medium">Total Bits</p>
                      <p className="text-2xl font-bold text-blue-900">{results.total_bits}</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                  <div className="flex items-center">
                    <TrendingUp className="h-8 w-8 text-purple-600 mr-3" />
                    <div>
                      <p className="text-sm text-purple-600 font-medium">Avg Confidence</p>
                      <p className="text-2xl font-bold text-purple-900">{formatNumber(results.avg_confidence)}</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                  <div className="flex items-center">
                    <Zap className="h-8 w-8 text-orange-600 mr-3" />
                    <div>
                      <p className="text-sm text-orange-600 font-medium">High Confidence</p>
                      <p className="text-2xl font-bold text-orange-900">{formatNumber(results.high_conf_percentage, 1)}%</p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 mb-2">Dataset Composition</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Cipher Streams:</span>
                      <span className="font-medium">{results.cipher_streams}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Random Streams:</span>
                      <span className="font-medium">{results.random_streams}</span>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 mb-2">Analysis Metrics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Confidence Correlation:</span>
                      <span className="font-medium">{formatNumber(results.confidence_correlation, 3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">High Conf. Predictions:</span>
                      <span className="font-medium">{results.high_conf_predictions}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* SHAP Visualizations */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                <BarChart3 className="h-6 w-6 mr-3 text-blue-600" />
                {results.model_name} SHAP Explainability Visualizations
              </h2>
              
              <div className="space-y-8">
                {/* Summary Plot */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">SHAP Summary Plot</h3>
                  <div className="flex justify-center">
                    <img 
                      src={`data:image/png;base64,${results.plots.summary_plot}`} 
                      alt="SHAP Summary Plot"
                      className="max-w-full h-auto rounded-lg shadow-md"
                    />
                  </div>
                  <p className="text-sm text-gray-600 mt-3 text-center">
                    Shows the most important features and their impact on {results.model_name} predictions
                  </p>
                </div>

                {/* Feature Importance Plot */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance</h3>
                  <div className="flex justify-center">
                    <img 
                      src={`data:image/png;base64,${results.plots.importance_plot}`} 
                      alt="Feature Importance Plot"
                      className="max-w-full h-auto rounded-lg shadow-md"
                    />
                  </div>
                  <p className="text-sm text-gray-600 mt-3 text-center">
                    Bar plot showing mean absolute SHAP values for top features in {results.model_name}
                  </p>
                </div>

                {/* Comparison Plot */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">{results.model_name} vs SHAP Importance</h3>
                  <div className="flex justify-center">
                    <img 
                      src={`data:image/png;base64,${results.plots.comparison_plot}`} 
                      alt="Comparison Plot"
                      className="max-w-full h-auto rounded-lg shadow-md"
                    />
                  </div>
                  <p className="text-sm text-gray-600 mt-3 text-center">
                    Comparison between {results.model_name} feature importance and SHAP importance
                  </p>
                </div>

                {/* Confidence Analysis Plot */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Confidence Analysis</h3>
                  <div className="flex justify-center">
                    <img 
                      src={`data:image/png;base64,${results.plots.confidence_plot}`} 
                      alt="Confidence Analysis Plot"
                      className="max-w-full h-auto rounded-lg shadow-md"
                    />
                  </div>
                  <p className="text-sm text-gray-600 mt-3 text-center">
                    Analysis of {results.model_name} prediction confidence and its relationship with SHAP values
                  </p>
                </div>

                {/* Bit Importance Plot */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Important Bit Positions</h3>
                  <div className="flex justify-center">
                    <img 
                      src={`data:image/png;base64,${results.plots.bit_plot}`} 
                      alt="Bit Importance Plot"
                      className="max-w-full h-auto rounded-lg shadow-md"
                    />
                  </div>
                  <p className="text-sm text-gray-600 mt-3 text-center">
                    Shows which bit positions are most discriminative for cipher detection using {results.model_name}
                  </p>
                </div>
              </div>
            </div>

            {/* Individual Predictions Analysis */}
            {results.individual_examples && results.individual_examples.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
                <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <Brain className="h-6 w-6 mr-3 text-purple-600" />
                  Individual Prediction Examples ({results.model_name})
                </h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {results.individual_examples.map((example, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <h3 className="text-lg font-semibold text-gray-900 mb-3">{example.type}</h3>
                      <div className="space-y-2 mb-4">
                        <div className="flex justify-between">
                          <span className="text-gray-600">True Label:</span>
                          <span className={`font-medium ${example.true_label === 'Cipher' ? 'text-red-600' : 'text-blue-600'}`}>
                            {example.true_label}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Predicted Probability (Cipher):</span>
                          <span className="font-medium">{formatNumber(example.predicted_prob)}</span>
                        </div>
                      </div>
                      
                      <h4 className="font-medium text-gray-900 mb-2">Top Contributing Features:</h4>
                      <div className="space-y-1 max-h-48 overflow-y-auto">
                        {example.top_features.map((feature, featureIndex) => (
                          <div key={featureIndex} className="flex justify-between items-center text-sm">
                            <span className="text-gray-600">{featureIndex + 1}. {feature.feature}:</span>
                            <span className={`font-medium ${feature.contribution > 0 ? 'text-red-600' : 'text-blue-600'}`}>
                              {feature.contribution > 0 ? '+' : ''}{formatNumber(feature.contribution)} {feature.direction}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top Bit Analysis */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                <TrendingUp className="h-6 w-6 mr-3 text-green-600" />
                Top Most Important Bit Positions ({results.model_name})
              </h2>
              
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Rank
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Bit Position
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        SHAP Importance
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Cipher Avg
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Random Avg
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {results.top_bits.slice(0, 10).map((bit, index) => (
                      <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {bit.rank}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          bit_{bit.bit_position}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatNumber(bit.importance)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-medium">
                          {formatNumber(bit.cipher_avg, 3)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-600 font-medium">
                          {formatNumber(bit.random_avg, 3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Model-Specific Key Insights */}
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl shadow-lg p-6 border border-blue-200">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                <Zap className="h-6 w-6 mr-3 text-yellow-600" />
                Key Insights for {results.model_name}
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-1 flex-shrink-0" />
                    <p className="text-gray-700">
                      <strong>{results.model_name} achieved</strong> {formatNumber(results.accuracy)} accuracy on cipher detection
                    </p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-1 flex-shrink-0" />
                    <p className="text-gray-700">
                      <strong>SHAP values reveal</strong> which bit positions are most important for {results.model_name} classification
                    </p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-1 flex-shrink-0" />
                    <p className="text-gray-700">
                      <strong>Individual analysis</strong> shows how each bit contributes to specific {results.model_name} predictions
                    </p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-1 flex-shrink-0" />
                    <p className="text-gray-700">
                      <strong>Confidence analysis</strong> helps identify reliable vs uncertain {results.model_name} predictions
                    </p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-1 flex-shrink-0" />
                    <p className="text-gray-700">
                      <strong>This explainability</strong> validates that {results.model_name} learns meaningful cipher patterns
                    </p>
                  </div>
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-1 flex-shrink-0" />
                    <p className="text-gray-700">
                      <strong>High confidence predictions:</strong> {formatNumber(results.high_conf_percentage, 1)}% of all predictions
                    </p>
                  </div>
                </div>
              </div>

              {/* Model-specific strengths */}
              <div className="mt-6 p-4 bg-white rounded-lg border border-blue-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                  <div className={`p-2 bg-gradient-to-r ${getModelColor(selectedModel)} rounded-lg mr-3`}>
                    {getModelIcon(selectedModel)}
                  </div>
                  {results.model_name} Advantages
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {availableModels[selectedModel]?.strengths?.map((strength, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                      <span className="text-sm text-gray-700">{strength}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Analysis Completion */}
            <div className="bg-green-50 border border-green-200 rounded-xl p-6">
              <div className="flex items-center">
                <CheckCircle className="h-8 w-8 text-green-600 mr-4" />
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-green-900">
                    {results.model_name} Analysis Complete!
                  </h3>
                  <p className="text-green-700">
                    Finished at: {new Date(results.finished_at).toLocaleString()}
                  </p>
                </div>
                <div className={`p-3 bg-gradient-to-r ${getModelColor(selectedModel)} rounded-full`}>
                  {getModelIcon(selectedModel)}
                </div>
              </div>
              
              {/* Quick Stats Summary */}
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-900">{formatNumber(results.accuracy)}</div>
                  <div className="text-sm text-green-600">Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-900">{results.total_bits}</div>
                  <div className="text-sm text-green-600">Features</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-900">{formatNumber(results.avg_confidence, 2)}</div>
                  <div className="text-sm text-green-600">Avg Confidence</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-900">{formatNumber(results.high_conf_percentage, 1)}%</div>
                  <div className="text-sm text-green-600">High Confidence</div>
                </div>
              </div>
            </div>

            {/* Compare Models Suggestion */}
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6">
              <div className="flex items-start space-x-3">
                <Info className="h-6 w-6 text-yellow-600 mt-1 flex-shrink-0" />
                <div>
                  <h3 className="text-lg font-semibold text-yellow-900 mb-2">
                    Want to Compare Different Models?
                  </h3>
                  <p className="text-yellow-800 mb-4">
                    Try running the same analysis with different machine learning models to compare their performance and interpretability on your cipher detection task.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(availableModels).filter(([key]) => key !== selectedModel).map(([key, model]) => (
                      <button
                        key={key}
                        onClick={() => {
                          setSelectedModel(key);
                          window.scrollTo({ top: 0, behavior: 'smooth' });
                        }}
                        className={`inline-flex items-center px-3 py-2 bg-gradient-to-r ${getModelColor(key)} text-white text-sm rounded-lg hover:shadow-md transition-all`}
                      >
                        {getModelIcon(key)}
                        <span className="ml-2">Try {model.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;