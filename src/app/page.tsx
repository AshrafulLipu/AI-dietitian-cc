import React from 'react';
import { Sparkles, TrendingUp, Shield, Zap, Camera, Brain, Heart, ArrowRight, Target, Activity, Apple, BarChart3, User, CheckCircle } from 'lucide-react';
import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">     
      {/* Main Content */}
      <main className="pt-24 pb-20 px-4 sm:px-6">
        <div className="max-w-6xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-20">
            <div className="inline-flex items-center space-x-2 bg-green-100 text-green-700 px-4 py-2 rounded-full text-sm font-medium mb-6 animate-pulse">
              <Sparkles className="w-4 h-4" />
              <span>AI-Powered Personalized Nutrition</span>
            </div>
            <h1 className="text-5xl sm:text-7xl font-bold text-gray-900 mb-6 leading-tight">
              Your Personal AI
              <span className="block bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                Nutrition Coach
              </span>
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-10 leading-relaxed">
              Smart Dietitian learns about your body, health goals, and medical conditions to provide personalized nutrition recommendations. Upload a food photo and get instant AI-powered insights tailored just for you.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/classifier" className="group px-8 py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white font-semibold rounded-xl hover:shadow-2xl hover:scale-105 transition-all flex items-center space-x-2">
                <span>Start Analyzing</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link href="/how-it-works" className="px-8 py-4 bg-white text-gray-700 font-semibold rounded-xl hover:shadow-lg transition-all border-2 border-gray-200">
                Learn More
              </Link>
            </div>
          </div>

          {/* Stats Section */}
          <div className="grid md:grid-cols-4 gap-6 mb-20">
            <div className="bg-white rounded-2xl p-6 shadow-lg text-center transform hover:scale-105 transition-transform">
              <div className="text-4xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent mb-2">
                86.96%
              </div>
              <p className="text-gray-600 font-medium text-sm">AI Accuracy</p>
            </div>
            <div className="bg-white rounded-2xl p-6 shadow-lg text-center transform hover:scale-105 transition-transform">
              <div className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">
                34+
              </div>
              <p className="text-gray-600 font-medium text-sm">Foods Recognized</p>
            </div>
            <div className="bg-white rounded-2xl p-6 shadow-lg text-center transform hover:scale-105 transition-transform">
              <div className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">
                &lt;2s
              </div>
              <p className="text-gray-600 font-medium text-sm">Analysis Time</p>
            </div>
            <div className="bg-white rounded-2xl p-6 shadow-lg text-center transform hover:scale-105 transition-transform">
              <div className="text-4xl font-bold bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent mb-2">
                100%
              </div>
              <p className="text-gray-600 font-medium text-sm">Personalized</p>
            </div>
          </div>

          {/* Key Features Highlight */}
          <div className="bg-gradient-to-br from-green-600 to-emerald-600 rounded-3xl p-12 mb-20 text-white shadow-2xl">
            <div className="text-center mb-10">
              <h2 className="text-4xl font-bold mb-4">Complete Nutrition Management</h2>
              <p className="text-xl text-green-50 max-w-2xl mx-auto">
                From personalized profiles to daily tracking, Smart Dietitian handles everything
              </p>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <Target className="w-10 h-10 mb-4" />
                <h3 className="text-xl font-bold mb-2">Personal Health Profile</h3>
                <p className="text-green-50">AI calculates your BMI and daily nutrition targets based on your age, height, weight, and health conditions</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <Brain className="w-10 h-10 mb-4" />
                <h3 className="text-xl font-bold mb-2">Smart Food Analysis</h3>
                <p className="text-green-50">Get personalized recommendations comparing each food against your goals and medical conditions</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                <BarChart3 className="w-10 h-10 mb-4" />
                <h3 className="text-xl font-bold mb-2">Daily Progress Tracking</h3>
                <p className="text-green-50">Monitor your calorie, protein, carbs, and fat intake throughout the day to stay on target</p>
              </div>
            </div>
          </div>

          {/* How It Works Section */}
          <div className="mb-20">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-gray-900 mb-4">Your Journey with Smart Dietitian</h2>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                Five simple steps to achieve your nutrition goals
              </p>
            </div>
            <div className="grid md:grid-cols-5 gap-4">
              <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow relative">
                <div className="absolute -top-4 -right-4 w-10 h-10 bg-green-600 text-white rounded-full flex items-center justify-center font-bold text-lg shadow-lg">
                  1
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl flex items-center justify-center mb-4">
                  <User className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">Sign Up</h3>
                <p className="text-sm text-gray-600">
                  Create your profile with health details and conditions
                </p>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow relative">
                <div className="absolute -top-4 -right-4 w-10 h-10 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-lg shadow-lg">
                  2
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center mb-4">
                  <Activity className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">Get Targets</h3>
                <p className="text-sm text-gray-600">
                  AI calculates your daily nutrition needs
                </p>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow relative">
                <div className="absolute -top-4 -right-4 w-10 h-10 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold text-lg shadow-lg">
                  3
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mb-4">
                  <Camera className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">Upload Food</h3>
                <p className="text-sm text-gray-600">
                  Take a photo of your meal
                </p>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow relative">
                <div className="absolute -top-4 -right-4 w-10 h-10 bg-orange-600 text-white rounded-full flex items-center justify-center font-bold text-lg shadow-lg">
                  4
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-500 rounded-xl flex items-center justify-center mb-4">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">AI Analysis</h3>
                <p className="text-sm text-gray-600">
                  Get personalized recommendations
                </p>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow relative">
                <div className="absolute -top-4 -right-4 w-10 h-10 bg-teal-600 text-white rounded-full flex items-center justify-center font-bold text-lg shadow-lg">
                  5
                </div>
                <div className="w-12 h-12 bg-gradient-to-br from-teal-500 to-green-500 rounded-xl flex items-center justify-center mb-4">
                  <CheckCircle className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">Track Progress</h3>
                <p className="text-sm text-gray-600">
                  Monitor your daily intake goals
                </p>
              </div>
            </div>
          </div>

          {/* Features Section */}
          <div className="mb-20">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-gray-900 mb-4">Powerful Features for Your Health</h2>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                Everything you need to make informed nutrition decisions
              </p>
            </div>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all group">
                <div className="w-14 h-14 bg-green-100 rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform">
                  <Target className="w-7 h-7 text-green-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">Personalized BMI & Goals</h3>
                <p className="text-gray-600 leading-relaxed">
                  Our AI analyzes your body metrics and health conditions to set precise daily targets for calories, protein, carbs, and fats tailored to your unique needs.
                </p>
              </div>
              <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all group">
                <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform">
                  <Apple className="w-7 h-7 text-blue-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">Medical Condition Aware</h3>
                <p className="text-gray-600 leading-relaxed">
                  Get food recommendations that consider your specific health conditions, with detailed pros and cons for each meal based on your medical profile.
                </p>
              </div>
              <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all group">
                <div className="w-14 h-14 bg-purple-100 rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform">
                  <BarChart3 className="w-7 h-7 text-purple-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">Real-Time Tracking</h3>
                <p className="text-gray-600 leading-relaxed">
                  Track every meal you consume and see your daily progress toward nutritional goals. Save or discard foods and maintain complete control over your diet.
                </p>
              </div>
              <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all group">
                <div className="w-14 h-14 bg-orange-100 rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform">
                  <Zap className="w-7 h-7 text-orange-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">Instant Food Recognition</h3>
                <p className="text-gray-600 leading-relaxed">
                  Our trained classification model identifies foods from photos in seconds, showing complete nutritional content before you decide to eat.
                </p>
              </div>
              <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all group">
                <div className="w-14 h-14 bg-pink-100 rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform">
                  <Heart className="w-7 h-7 text-pink-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">Smart Recommendations</h3>
                <p className="text-gray-600 leading-relaxed">
                  Receive AI-powered advice on whether each food suits your goals and health conditions, with clear explanations of benefits and concerns.
                </p>
              </div>
              <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all group">
                <div className="w-14 h-14 bg-teal-100 rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform">
                  <Shield className="w-7 h-7 text-teal-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">Privacy & Security</h3>
                <p className="text-gray-600 leading-relaxed">
                  Your health data and food photos are processed securely. We prioritize your privacy while delivering personalized nutrition insights.
                </p>
              </div>
            </div>
          </div>

          {/* CTA Section */}
          <div className="bg-gradient-to-r from-green-600 to-emerald-600 rounded-3xl p-12 text-center shadow-2xl">
            <h2 className="text-4xl font-bold text-white mb-4">
              Start Your Personalized Nutrition Journey
            </h2>
            <p className="text-xl text-green-50 mb-8 max-w-2xl mx-auto">
              Join Smart Dietitian today and get AI-powered recommendations tailored to your body, health, and goals
            </p>
            <Link href="/classifier" className="inline-flex items-center space-x-2 px-8 py-4 bg-white text-green-600 font-bold rounded-xl hover:shadow-2xl hover:scale-105 transition-all">
              <span>Try Smart Dietitian Now</span>
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}