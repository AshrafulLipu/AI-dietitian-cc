// src/app/(routes)/food-analysis/page.tsx
'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter, useSearchParams } from 'next/navigation';
import { 
  Loader2, ThumbsUp, ThumbsDown, TrendingUp, AlertTriangle,
  CheckCircle, XCircle, ArrowLeft, Utensils, Activity, Info,
  Flame, Drumstick, Wheat, Droplet, PieChart
} from 'lucide-react';

interface AnalysisData {
  foodName: string;
  nutritionalData: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
  analysis: {
    recommendation: 'recommended' | 'moderate' | 'not_recommended';
    pros: string[];
    cons: string[];
    summary: string;
  };
  userProfile: {
    dailyCalories: number;
    dailyProtein: number;
    dailyCarbs: number;
    dailyFat: number;
  };
  currentConsumption?: {
    totalCalories: number;
    totalProtein: number;
    totalCarbs: number;
    totalFat: number;
    mealsCount: number;
  };
  afterConsumption?: {
    totalCalories: number;
    totalProtein: number;
    totalCarbs: number;
    totalFat: number;
  };
}

// Separate component that uses useSearchParams
function FoodAnalysisContent() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const searchParams = useSearchParams();
  const foodName = searchParams.get('food');

  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/signin');
      return;
    }

    if (!foodName) {
      router.push('/classifier');
      return;
    }

    analyzeFood();
  }, [foodName, status]);

  const analyzeFood = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/food/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ foodName }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze food');
      }

      setAnalysisData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleEat = async () => {
    if (!analysisData) return;

    try {
      setActionLoading(true);

      const response = await fetch('/api/food/log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          foodName: analysisData.foodName,
          calories: analysisData.nutritionalData.calories,
          protein: analysisData.nutritionalData.protein,
          carbs: analysisData.nutritionalData.carbs,
          fat: analysisData.nutritionalData.fat,
          recommendation: analysisData.analysis.recommendation,
          pros: analysisData.analysis.pros,
          cons: analysisData.analysis.cons,
          aiAnalysis: analysisData.analysis.summary,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to log food');
      }

      router.push('/profile?logged=true');
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to log food');
    } finally {
      setActionLoading(false);
    }
  };

  const handleDontEat = () => {
    router.push('/classifier');
  };

  if (status === 'loading' || loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-emerald-600 animate-spin mx-auto mb-4" />
          <p className="text-emerald-700 font-semibold">Analyzing your food...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Analysis Failed</h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={() => router.push('/classifier')}
            className="bg-emerald-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-emerald-700 transition"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!analysisData) return null;

  const getRecommendationColor = () => {
    switch (analysisData.analysis.recommendation) {
      case 'recommended':
        return 'from-green-500 to-emerald-600';
      case 'moderate':
        return 'from-yellow-500 to-orange-500';
      case 'not_recommended':
        return 'from-red-500 to-pink-600';
      default:
        return 'from-gray-500 to-gray-600';
    }
  };

  const getRecommendationIcon = () => {
    switch (analysisData.analysis.recommendation) {
      case 'recommended':
        return <CheckCircle className="w-12 h-12" />;
      case 'moderate':
        return <AlertTriangle className="w-12 h-12" />;
      case 'not_recommended':
        return <XCircle className="w-12 h-12" />;
    }
  };

  const getRecommendationText = () => {
    switch (analysisData.analysis.recommendation) {
      case 'recommended':
        return 'Recommended';
      case 'moderate':
        return 'Eat in Moderation';
      case 'not_recommended':
        return 'Not Recommended';
    }
  };

  const calculatePercentage = (value: number, target: number) => {
    return ((value / target) * 100).toFixed(1);
  };

  const getStatusColor = (percentage: number) => {
    if (percentage >= 100) return 'text-red-600';
    if (percentage >= 80) return 'text-orange-600';
    if (percentage >= 50) return 'text-yellow-600';
    return 'text-green-600';
  };

  const currentConsumption = analysisData.currentConsumption;
  const afterConsumption = analysisData.afterConsumption;

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 py-8 px-4">
      <div className="max-w-4xl mx-auto mt-12">
        {/* Food Name Header */}
        <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-8 mb-6">
          <div className="flex items-center gap-4 mb-4">
            <div className="bg-gradient-to-br from-emerald-100 to-green-100 p-4 rounded-2xl">
              <Utensils className="w-8 h-8 text-emerald-600" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800">{analysisData.foodName}</h1>
              <p className="text-gray-600">AI-Powered Nutritional Analysis</p>
            </div>
          </div>

          {/* Recommendation Badge */}
          <div className={`bg-gradient-to-r ${getRecommendationColor()} text-white p-6 rounded-2xl flex items-center gap-4`}>
            {getRecommendationIcon()}
            <div>
              <p className="text-sm font-medium opacity-90">AI Recommendation</p>
              <p className="text-2xl font-bold">{getRecommendationText()}</p>
            </div>
          </div>
        </div>

        {/* Current Consumption Context */}
        {currentConsumption && (
          <div className="bg-white rounded-3xl shadow-xl border border-blue-200 p-8 mb-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Activity className="w-6 h-6 text-blue-600" />
              Your Today's Consumption
            </h2>
            <p className="text-gray-600 mb-6">
              You've already consumed {currentConsumption.mealsCount} meal{currentConsumption.mealsCount !== 1 ? 's' : ''} today. 
              Here's where you stand:
            </p>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-xl border-2 border-orange-200">
                <div className="flex items-center gap-2 mb-2">
                  <Flame className="w-5 h-5 text-orange-600" />
                  <p className="text-sm text-orange-700 font-medium">Calories</p>
                </div>
                <p className="text-2xl font-bold text-orange-900">{currentConsumption.totalCalories.toFixed(0)}</p>
                <p className={`text-sm font-semibold mt-1 ${getStatusColor(parseFloat(calculatePercentage(currentConsumption.totalCalories, analysisData.userProfile.dailyCalories)))}`}>
                  {calculatePercentage(currentConsumption.totalCalories, analysisData.userProfile.dailyCalories)}% consumed
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-xl border-2 border-purple-200">
                <div className="flex items-center gap-2 mb-2">
                  <Drumstick className="w-5 h-5 text-purple-600" />
                  <p className="text-sm text-purple-700 font-medium">Protein</p>
                </div>
                <p className="text-2xl font-bold text-purple-900">{currentConsumption.totalProtein.toFixed(0)}g</p>
                <p className={`text-sm font-semibold mt-1 ${getStatusColor(parseFloat(calculatePercentage(currentConsumption.totalProtein, analysisData.userProfile.dailyProtein)))}`}>
                  {calculatePercentage(currentConsumption.totalProtein, analysisData.userProfile.dailyProtein)}% consumed
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-amber-50 to-amber-100 p-4 rounded-xl border-2 border-amber-200">
                <div className="flex items-center gap-2 mb-2">
                  <Wheat className="w-5 h-5 text-amber-600" />
                  <p className="text-sm text-amber-700 font-medium">Carbs</p>
                </div>
                <p className="text-2xl font-bold text-amber-900">{currentConsumption.totalCarbs.toFixed(0)}g</p>
                <p className={`text-sm font-semibold mt-1 ${getStatusColor(parseFloat(calculatePercentage(currentConsumption.totalCarbs, analysisData.userProfile.dailyCarbs)))}`}>
                  {calculatePercentage(currentConsumption.totalCarbs, analysisData.userProfile.dailyCarbs)}% consumed
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-xl border-2 border-blue-200">
                <div className="flex items-center gap-2 mb-2">
                  <Droplet className="w-5 h-5 text-blue-600" />
                  <p className="text-sm text-blue-700 font-medium">Fat</p>
                </div>
                <p className="text-2xl font-bold text-blue-900">{currentConsumption.totalFat.toFixed(0)}g</p>
                <p className={`text-sm font-semibold mt-1 ${getStatusColor(parseFloat(calculatePercentage(currentConsumption.totalFat, analysisData.userProfile.dailyFat)))}`}>
                  {calculatePercentage(currentConsumption.totalFat, analysisData.userProfile.dailyFat)}% consumed
                </p>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-800">
                <p className="font-semibold mb-1">Remaining Allowance:</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                  <span><strong>Calories:</strong> {(analysisData.userProfile.dailyCalories - currentConsumption.totalCalories).toFixed(0)} kcal</span>
                  <span><strong>Protein:</strong> {(analysisData.userProfile.dailyProtein - currentConsumption.totalProtein).toFixed(1)}g</span>
                  <span><strong>Carbs:</strong> {(analysisData.userProfile.dailyCarbs - currentConsumption.totalCarbs).toFixed(1)}g</span>
                  <span><strong>Fat:</strong> {(analysisData.userProfile.dailyFat - currentConsumption.totalFat).toFixed(1)}g</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* This Food's Nutritional Information */}
        <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-8 mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
            <PieChart className="w-6 h-6 text-emerald-600" />
            This Food's Nutrition
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-xl border border-blue-200">
              <p className="text-sm text-blue-700 font-medium mb-1">Calories</p>
              <p className="text-2xl font-bold text-blue-900">{analysisData.nutritionalData.calories}</p>
              <p className="text-xs text-blue-600 mt-1">
                {calculatePercentage(analysisData.nutritionalData.calories, analysisData.userProfile.dailyCalories)}% of daily target
              </p>
            </div>
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-xl border border-purple-200">
              <p className="text-sm text-purple-700 font-medium mb-1">Protein</p>
              <p className="text-2xl font-bold text-purple-900">{analysisData.nutritionalData.protein}g</p>
              <p className="text-xs text-purple-600 mt-1">
                {calculatePercentage(analysisData.nutritionalData.protein, analysisData.userProfile.dailyProtein)}% of daily target
              </p>
            </div>
            <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-xl border border-orange-200">
              <p className="text-sm text-orange-700 font-medium mb-1">Carbs</p>
              <p className="text-2xl font-bold text-orange-900">{analysisData.nutritionalData.carbs}g</p>
              <p className="text-xs text-orange-600 mt-1">
                {calculatePercentage(analysisData.nutritionalData.carbs, analysisData.userProfile.dailyCarbs)}% of daily target
              </p>
            </div>
            <div className="bg-gradient-to-br from-pink-50 to-pink-100 p-4 rounded-xl border border-pink-200">
              <p className="text-sm text-pink-700 font-medium mb-1">Fat</p>
              <p className="text-2xl font-bold text-pink-900">{analysisData.nutritionalData.fat}g</p>
              <p className="text-xs text-pink-600 mt-1">
                {calculatePercentage(analysisData.nutritionalData.fat, analysisData.userProfile.dailyFat)}% of daily target
              </p>
            </div>
          </div>
        </div>

        {/* After Eating Projection */}
        {afterConsumption && currentConsumption && (
          <div className="bg-white rounded-3xl shadow-xl border border-purple-200 p-8 mb-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-purple-600" />
              If You Eat This...
            </h2>
            <p className="text-gray-600 mb-6">Your daily totals would become:</p>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className={`p-4 rounded-xl border-2 ${
                afterConsumption.totalCalories > analysisData.userProfile.dailyCalories 
                  ? 'bg-red-50 border-red-300' 
                  : afterConsumption.totalCalories > analysisData.userProfile.dailyCalories * 0.8
                  ? 'bg-orange-50 border-orange-300'
                  : 'bg-green-50 border-green-300'
              }`}>
                <p className="text-sm font-medium mb-1 text-gray-700">Total Calories</p>
                <p className="text-2xl font-bold text-gray-900">{afterConsumption.totalCalories.toFixed(0)}</p>
                <p className={`text-sm font-semibold mt-1 ${getStatusColor(parseFloat(calculatePercentage(afterConsumption.totalCalories, analysisData.userProfile.dailyCalories)))}`}>
                  {calculatePercentage(afterConsumption.totalCalories, analysisData.userProfile.dailyCalories)}% of target
                </p>
                {afterConsumption.totalCalories > analysisData.userProfile.dailyCalories && (
                  <p className="text-xs text-red-700 font-semibold mt-1">⚠️ OVER LIMIT</p>
                )}
              </div>
              
              <div className={`p-4 rounded-xl border-2 ${
                afterConsumption.totalProtein > analysisData.userProfile.dailyProtein 
                  ? 'bg-red-50 border-red-300' 
                  : afterConsumption.totalProtein > analysisData.userProfile.dailyProtein * 0.8
                  ? 'bg-orange-50 border-orange-300'
                  : 'bg-green-50 border-green-300'
              }`}>
                <p className="text-sm font-medium mb-1 text-gray-700">Total Protein</p>
                <p className="text-2xl font-bold text-gray-900">{afterConsumption.totalProtein.toFixed(0)}g</p>
                <p className={`text-sm font-semibold mt-1 ${getStatusColor(parseFloat(calculatePercentage(afterConsumption.totalProtein, analysisData.userProfile.dailyProtein)))}`}>
                  {calculatePercentage(afterConsumption.totalProtein, analysisData.userProfile.dailyProtein)}% of target
                </p>
                {afterConsumption.totalProtein > analysisData.userProfile.dailyProtein && (
                  <p className="text-xs text-red-700 font-semibold mt-1">⚠️ OVER LIMIT</p>
                )}
              </div>
              
              <div className={`p-4 rounded-xl border-2 ${
                afterConsumption.totalCarbs > analysisData.userProfile.dailyCarbs 
                  ? 'bg-red-50 border-red-300' 
                  : afterConsumption.totalCarbs > analysisData.userProfile.dailyCarbs * 0.8
                  ? 'bg-orange-50 border-orange-300'
                  : 'bg-green-50 border-green-300'
              }`}>
                <p className="text-sm font-medium mb-1 text-gray-700">Total Carbs</p>
                <p className="text-2xl font-bold text-gray-900">{afterConsumption.totalCarbs.toFixed(0)}g</p>
                <p className={`text-sm font-semibold mt-1 ${getStatusColor(parseFloat(calculatePercentage(afterConsumption.totalCarbs, analysisData.userProfile.dailyCarbs)))}`}>
                  {calculatePercentage(afterConsumption.totalCarbs, analysisData.userProfile.dailyCarbs)}% of target
                </p>
                {afterConsumption.totalCarbs > analysisData.userProfile.dailyCarbs && (
                  <p className="text-xs text-red-700 font-semibold mt-1">⚠️ OVER LIMIT</p>
                )}
              </div>
              
              <div className={`p-4 rounded-xl border-2 ${
                afterConsumption.totalFat > analysisData.userProfile.dailyFat 
                  ? 'bg-red-50 border-red-300' 
                  : afterConsumption.totalFat > analysisData.userProfile.dailyFat * 0.8
                  ? 'bg-orange-50 border-orange-300'
                  : 'bg-green-50 border-green-300'
              }`}>
                <p className="text-sm font-medium mb-1 text-gray-700">Total Fat</p>
                <p className="text-2xl font-bold text-gray-900">{afterConsumption.totalFat.toFixed(0)}g</p>
                <p className={`text-sm font-semibold mt-1 ${getStatusColor(parseFloat(calculatePercentage(afterConsumption.totalFat, analysisData.userProfile.dailyFat)))}`}>
                  {calculatePercentage(afterConsumption.totalFat, analysisData.userProfile.dailyFat)}% of target
                </p>
                {afterConsumption.totalFat > analysisData.userProfile.dailyFat && (
                  <p className="text-xs text-red-700 font-semibold mt-1">⚠️ OVER LIMIT</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* AI Analysis */}
        <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-8 mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">AI Analysis</h2>
          
          <div className="bg-gradient-to-r from-emerald-50 to-green-50 p-6 rounded-xl border border-emerald-200 mb-6">
            <p className="text-gray-700 leading-relaxed">{analysisData.analysis.summary}</p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Pros */}
            <div>
              <h3 className="text-lg font-bold text-green-700 mb-4 flex items-center gap-2">
                <ThumbsUp className="w-5 h-5" />
                Benefits
              </h3>
              <ul className="space-y-2">
                {analysisData.analysis.pros.map((pro, index) => (
                  <li key={index} className="flex items-start gap-2 text-gray-700">
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                    <span>{pro}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Cons */}
            <div>
              <h3 className="text-lg font-bold text-red-700 mb-4 flex items-center gap-2">
                <ThumbsDown className="w-5 h-5" />
                Concerns
              </h3>
              <ul className="space-y-2">
                {analysisData.analysis.cons.map((con, index) => (
                  <li key={index} className="flex items-start gap-2 text-gray-700">
                    <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <span>{con}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="grid md:grid-cols-2 gap-4">
          <button
            onClick={handleEat}
            disabled={actionLoading}
            className="bg-gradient-to-r from-green-500 to-emerald-600 text-white py-4 px-6 rounded-xl font-bold text-lg shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-2"
          >
            {actionLoading ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                Logging...
              </>
            ) : (
              <>
                <CheckCircle className="w-6 h-6" />
                I'll Eat This
              </>
            )}
          </button>
          <button
            onClick={handleDontEat}
            disabled={actionLoading}
            className="bg-gradient-to-r from-gray-500 to-gray-600 text-white py-4 px-6 rounded-xl font-bold text-lg shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-2"
          >
            <XCircle className="w-6 h-6" />
            I Won't Eat This
          </button>
        </div>
      </div>
    </div>
  );
}

// Main page component with Suspense boundary
export default function FoodAnalysisPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-emerald-600 animate-spin mx-auto mb-4" />
          <p className="text-emerald-700 font-semibold">Loading...</p>
        </div>
      </div>
    }>
      <FoodAnalysisContent />
    </Suspense>
  );
}
