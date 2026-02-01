// // src/app/(routes)/profile/page.tsx
// 'use client';

// import { useState, useEffect } from 'react';
// import { useSession } from 'next-auth/react';
// import { useRouter, useSearchParams } from 'next/navigation';
// import {
//   User, Activity, TrendingUp, Calendar, Flame, Drumstick, Wheat, Droplet,
//   Clock, CheckCircle, AlertCircle, Loader2, AlertTriangle, Trophy, Target,
//   TrendingDown, Info
// } from 'lucide-react';

// interface UserData {
//   name: string;
//   email: string;
//   age: number;
//   height: number;
//   weight: number;
//   gender: string;
//   bmi: number;
//   medicalCondition: string | null;
//   dailyCalories: number;
//   dailyProtein: number;
//   dailyCarbs: number;
//   dailyFat: number;
// }

// interface DailyNutrition {
//   totalCalories: number;
//   totalProtein: number;
//   totalCarbs: number;
//   totalFat: number;
//   mealsCount: number;
// }

// interface FoodLog {
//   id: number;
//   foodName: string;
//   calories: number;
//   protein: number;
//   carbs: number;
//   fat: number;
//   recommendation: string;
//   consumedAt: string;
// }

// export default function ProfilePage() {
//   const { data: session, status } = useSession();
//   const router = useRouter();
//   const searchParams = useSearchParams();
//   const showSuccess = searchParams.get('logged') === 'true';

//   const [userData, setUserData] = useState<UserData | null>(null);
//   const [dailyNutrition, setDailyNutrition] = useState<DailyNutrition>({
//     totalCalories: 0,
//     totalProtein: 0,
//     totalCarbs: 0,
//     totalFat: 0,
//     mealsCount: 0,
//   });
//   const [foodLogs, setFoodLogs] = useState<FoodLog[]>([]);
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     if (status === 'unauthenticated') {
//       router.push('/signin');
//       return;
//     }

//     if (session?.user?.email) {
//       fetchUserData();
//       fetchDailyNutrition();
//     }
//   }, [status, session]);

//   const fetchUserData = async () => {
//     try {
//       const response = await fetch('/api/user/profile');
//       const data = await response.json();
//       if (response.ok) {
//         setUserData(data.user);
//       }
//     } catch (error) {
//       console.error('Error fetching user data:', error);
//     }
//   };

//   const fetchDailyNutrition = async () => {
//     try {
//       setLoading(true);
//       const response = await fetch('/api/food/log');
//       const data = await response.json();
      
//       if (response.ok) {
//         setDailyNutrition(data.dailyNutrition);
//         setFoodLogs(data.foodLogs);
//       }
//     } catch (error) {
//       console.error('Error fetching daily nutrition:', error);
//     } finally {
//       setLoading(false);
//     }
//   };

//   if (status === 'loading' || loading) {
//     return (
//       <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 flex items-center justify-center">
//         <div className="text-center">
//           <Loader2 className="w-12 h-12 text-emerald-600 animate-spin mx-auto mb-4" />
//           <p className="text-emerald-700 font-semibold">Loading profile...</p>
//         </div>
//       </div>
//     );
//   }

//   if (!userData) return null;

//   const calculateProgress = (current: number, target: number) => {
//     return Math.min((current / target) * 100, 100);
//   };

//   const calculatePercentage = (current: number, target: number) => {
//     return ((current / target) * 100).toFixed(1);
//   };

//   const getProgressColor = (percentage: number) => {
//     if (percentage < 50) return 'from-green-500 to-emerald-600';
//     if (percentage < 80) return 'from-yellow-500 to-orange-500';
//     if (percentage < 100) return 'from-orange-500 to-red-500';
//     return 'from-red-600 to-pink-600';
//   };

//   const getProgressStatus = (current: number, target: number) => {
//     const percentage = (current / target) * 100;
//     if (percentage >= 100) return { status: 'exceeded', icon: AlertTriangle, color: 'text-red-600', bgColor: 'bg-red-50', borderColor: 'border-red-200' };
//     if (percentage >= 80) return { status: 'warning', icon: AlertCircle, color: 'text-orange-600', bgColor: 'bg-orange-50', borderColor: 'border-orange-200' };
//     if (percentage >= 50) return { status: 'good', icon: TrendingUp, color: 'text-yellow-600', bgColor: 'bg-yellow-50', borderColor: 'border-yellow-200' };
//     return { status: 'excellent', icon: CheckCircle, color: 'text-green-600', bgColor: 'bg-green-50', borderColor: 'border-green-200' };
//   };

//   const getBMICategory = (bmi: number) => {
//     if (bmi < 18.5) return { text: 'Underweight', color: 'text-blue-600' };
//     if (bmi < 25) return { text: 'Normal', color: 'text-green-600' };
//     if (bmi < 30) return { text: 'Overweight', color: 'text-orange-600' };
//     return { text: 'Obese', color: 'text-red-600' };
//   };

//   const bmiCategory = getBMICategory(userData.bmi);

//   // Get overall status
//   const caloriesStatus = getProgressStatus(dailyNutrition.totalCalories, userData.dailyCalories);
//   const proteinStatus = getProgressStatus(dailyNutrition.totalProtein, userData.dailyProtein);
//   const carbsStatus = getProgressStatus(dailyNutrition.totalCarbs, userData.dailyCarbs);
//   const fatStatus = getProgressStatus(dailyNutrition.totalFat, userData.dailyFat);

//   const hasExceeded = [caloriesStatus, proteinStatus, carbsStatus, fatStatus].some(s => s.status === 'exceeded');
//   const hasWarning = [caloriesStatus, proteinStatus, carbsStatus, fatStatus].some(s => s.status === 'warning');

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 py-8 px-4">
//       <div className="max-w-7xl mx-auto mt-12">
//         {/* Success Message */}
//         {showSuccess && (
//           <div className="mb-6 bg-green-50 border border-green-200 text-green-800 px-6 py-4 rounded-xl flex items-center gap-3 animate-pulse">
//             <CheckCircle className="w-6 h-6 flex-shrink-0" />
//             <p className="font-semibold">Food logged successfully! Your daily nutrition has been updated.</p>
//           </div>
//         )}

//         {/* Alert Messages */}
//         {hasExceeded && (
//           <div className="mb-6 bg-red-50 border-2 border-red-300 text-red-800 px-6 py-4 rounded-xl flex items-start gap-3">
//             <AlertTriangle className="w-6 h-6 flex-shrink-0 mt-0.5" />
//             <div>
//               <p className="font-bold text-lg">⚠️ Daily Limit Exceeded!</p>
//               <p className="text-sm mt-1">You've exceeded your daily limit for one or more nutrients. Consider lighter meals for the rest of the day.</p>
//             </div>
//           </div>
//         )}

//         {!hasExceeded && hasWarning && (
//           <div className="mb-6 bg-orange-50 border-2 border-orange-300 text-orange-800 px-6 py-4 rounded-xl flex items-start gap-3">
//             <AlertCircle className="w-6 h-6 flex-shrink-0 mt-0.5" />
//             <div>
//               <p className="font-bold text-lg">⚠️ Approaching Daily Limit</p>
//               <p className="text-sm mt-1">You're getting close to your daily limits. Be mindful of your next meals.</p>
//             </div>
//           </div>
//         )}

//         {/* Header */}
//         <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-8 mb-6">
//           <div className="flex items-start justify-between flex-wrap gap-4">
//             <div className="flex items-center gap-4">
//               <div className="bg-gradient-to-br from-emerald-500 to-green-600 text-white w-20 h-20 rounded-2xl flex items-center justify-center">
//                 <User className="w-10 h-10" />
//               </div>
//               <div>
//                 <h1 className="text-3xl font-bold text-gray-800">{userData.name}</h1>
//                 <p className="text-gray-600">{userData.email}</p>
//                 <div className="flex items-center gap-4 mt-2 flex-wrap">
//                   <span className="text-sm bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full font-semibold">
//                     {userData.age} years old
//                   </span>
//                   <span className="text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-full font-semibold">
//                     {userData.gender}
//                   </span>
//                 </div>
//               </div>
//             </div>
            
//             {/* Daily Progress Summary */}
//             <div className="bg-gradient-to-br from-emerald-100 to-green-100 px-6 py-4 rounded-2xl border-2 border-emerald-300">
//               <div className="flex items-center gap-2 mb-2">
//                 <Trophy className="w-5 h-5 text-emerald-700" />
//                 <span className="font-bold text-emerald-900">Today's Status</span>
//               </div>
//               <p className="text-2xl font-bold text-emerald-800">{dailyNutrition.mealsCount} meals logged</p>
//             </div>
//           </div>
//         </div>

//         <div className="grid lg:grid-cols-3 gap-6 mb-6">
//           {/* Physical Stats */}
//           <div className="bg-white rounded-2xl shadow-xl border border-emerald-100 p-6">
//             <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
//               <Activity className="w-5 h-5 text-emerald-600" />
//               Physical Stats
//             </h2>
//             <div className="space-y-4">
//               <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
//                 <span className="text-gray-600 font-medium">Height</span>
//                 <span className="font-bold text-gray-800">{userData.height} cm</span>
//               </div>
//               <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
//                 <span className="text-gray-600 font-medium">Weight</span>
//                 <span className="font-bold text-gray-800">{userData.weight} kg</span>
//               </div>
//               <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
//                 <span className="text-gray-600 font-medium">BMI</span>
//                 <span className={`font-bold ${bmiCategory.color}`}>
//                   {userData.bmi.toFixed(1)} ({bmiCategory.text})
//                 </span>
//               </div>
//             </div>
//           </div>

//           {/* Daily Goals */}
//           <div className="bg-white rounded-2xl shadow-xl border border-emerald-100 p-6">
//             <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
//               <Target className="w-5 h-5 text-emerald-600" />
//               Daily Goals
//             </h2>
//             <div className="space-y-4">
//               <div className="flex justify-between items-center p-3 bg-gradient-to-r from-orange-50 to-orange-100 rounded-lg border border-orange-200">
//                 <span className="text-gray-700 font-medium">Calories</span>
//                 <span className="font-bold text-gray-800">{userData.dailyCalories} kcal</span>
//               </div>
//               <div className="flex justify-between items-center p-3 bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg border border-purple-200">
//                 <span className="text-gray-700 font-medium">Protein</span>
//                 <span className="font-bold text-gray-800">{userData.dailyProtein}g</span>
//               </div>
//               <div className="flex justify-between items-center p-3 bg-gradient-to-r from-amber-50 to-amber-100 rounded-lg border border-amber-200">
//                 <span className="text-gray-700 font-medium">Carbs</span>
//                 <span className="font-bold text-gray-800">{userData.dailyCarbs}g</span>
//               </div>
//               <div className="flex justify-between items-center p-3 bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg border border-blue-200">
//                 <span className="text-gray-700 font-medium">Fat</span>
//                 <span className="font-bold text-gray-800">{userData.dailyFat}g</span>
//               </div>
//             </div>
//           </div>

//           {/* Medical Info */}
//           <div className="bg-white rounded-2xl shadow-xl border border-emerald-100 p-6">
//             <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
//               <AlertCircle className="w-5 h-5 text-emerald-600" />
//               Medical Info
//             </h2>
//             <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-xl border border-blue-200 min-h-[120px] flex items-center">
//               {userData.medicalCondition ? (
//                 <p className="text-gray-700 font-medium">{userData.medicalCondition}</p>
//               ) : (
//                 <p className="text-gray-500 italic">No medical conditions reported</p>
//               )}
//             </div>
//           </div>
//         </div>

//         {/* Today's Progress */}
//         <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-8 mb-6">
//           <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
//             <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
//               <Calendar className="w-6 h-6 text-emerald-600" />
//               Today's Progress
//             </h2>
//             <div className="flex items-center gap-3">
//               <div className="bg-emerald-100 text-emerald-700 px-4 py-2 rounded-full font-semibold">
//                 {dailyNutrition.mealsCount} meals logged
//               </div>
//             </div>
//           </div>

//           <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
//             {/* Calories */}
//             <div className={`bg-gradient-to-br from-orange-50 to-orange-100 p-6 rounded-2xl border-2 ${caloriesStatus.borderColor} relative overflow-hidden`}>
//               <div className="absolute top-2 right-2">
//                 <caloriesStatus.icon className={`w-6 h-6 ${caloriesStatus.color}`} />
//               </div>
//               <div className="flex items-center gap-3 mb-3">
//                 <div className="bg-orange-500 text-white p-2 rounded-lg">
//                   <Flame className="w-5 h-5" />
//                 </div>
//                 <span className="font-bold text-gray-800">Calories</span>
//               </div>
//               <p className="text-3xl font-bold text-gray-900 mb-1">
//                 {dailyNutrition.totalCalories.toFixed(0)}
//               </p>
//               <p className="text-sm text-gray-600 mb-3">of {userData.dailyCalories} kcal</p>
              
//               <div className="bg-gray-200 h-3 rounded-full overflow-hidden mb-2">
//                 <div
//                   className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${getProgressColor(
//                     parseFloat(calculatePercentage(dailyNutrition.totalCalories, userData.dailyCalories))
//                   )}`}
//                   style={{
//                     width: `${Math.min(calculateProgress(dailyNutrition.totalCalories, userData.dailyCalories), 100)}%`,
//                   }}
//                 ></div>
//               </div>
              
//               <div className="flex items-center justify-between text-sm">
//                 <span className={`font-bold ${caloriesStatus.color}`}>
//                   {calculatePercentage(dailyNutrition.totalCalories, userData.dailyCalories)}%
//                 </span>
//                 <span className="text-gray-600">
//                   {(userData.dailyCalories - dailyNutrition.totalCalories).toFixed(0)} left
//                 </span>
//               </div>

//               {caloriesStatus.status === 'exceeded' && (
//                 <div className="mt-3 bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-semibold">
//                   LIMIT EXCEEDED
//                 </div>
//               )}
//               {caloriesStatus.status === 'warning' && (
//                 <div className="mt-3 bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-semibold">
//                   APPROACHING LIMIT
//                 </div>
//               )}
//             </div>

//             {/* Protein */}
//             <div className={`bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-2xl border-2 ${proteinStatus.borderColor} relative overflow-hidden`}>
//               <div className="absolute top-2 right-2">
//                 <proteinStatus.icon className={`w-6 h-6 ${proteinStatus.color}`} />
//               </div>
//               <div className="flex items-center gap-3 mb-3">
//                 <div className="bg-purple-500 text-white p-2 rounded-lg">
//                   <Drumstick className="w-5 h-5" />
//                 </div>
//                 <span className="font-bold text-gray-800">Protein</span>
//               </div>
//               <p className="text-3xl font-bold text-gray-900 mb-1">
//                 {dailyNutrition.totalProtein.toFixed(0)}g
//               </p>
//               <p className="text-sm text-gray-600 mb-3">of {userData.dailyProtein}g</p>
              
//               <div className="bg-gray-200 h-3 rounded-full overflow-hidden mb-2">
//                 <div
//                   className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${getProgressColor(
//                     parseFloat(calculatePercentage(dailyNutrition.totalProtein, userData.dailyProtein))
//                   )}`}
//                   style={{
//                     width: `${Math.min(calculateProgress(dailyNutrition.totalProtein, userData.dailyProtein), 100)}%`,
//                   }}
//                 ></div>
//               </div>
              
//               <div className="flex items-center justify-between text-sm">
//                 <span className={`font-bold ${proteinStatus.color}`}>
//                   {calculatePercentage(dailyNutrition.totalProtein, userData.dailyProtein)}%
//                 </span>
//                 <span className="text-gray-600">
//                   {(userData.dailyProtein - dailyNutrition.totalProtein).toFixed(1)}g left
//                 </span>
//               </div>

//               {proteinStatus.status === 'exceeded' && (
//                 <div className="mt-3 bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-semibold">
//                   LIMIT EXCEEDED
//                 </div>
//               )}
//               {proteinStatus.status === 'warning' && (
//                 <div className="mt-3 bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-semibold">
//                   APPROACHING LIMIT
//                 </div>
//               )}
//             </div>

//             {/* Carbs */}
//             <div className={`bg-gradient-to-br from-amber-50 to-amber-100 p-6 rounded-2xl border-2 ${carbsStatus.borderColor} relative overflow-hidden`}>
//               <div className="absolute top-2 right-2">
//                 <carbsStatus.icon className={`w-6 h-6 ${carbsStatus.color}`} />
//               </div>
//               <div className="flex items-center gap-3 mb-3">
//                 <div className="bg-amber-500 text-white p-2 rounded-lg">
//                   <Wheat className="w-5 h-5" />
//                 </div>
//                 <span className="font-bold text-gray-800">Carbs</span>
//               </div>
//               <p className="text-3xl font-bold text-gray-900 mb-1">
//                 {dailyNutrition.totalCarbs.toFixed(0)}g
//               </p>
//               <p className="text-sm text-gray-600 mb-3">of {userData.dailyCarbs}g</p>
              
//               <div className="bg-gray-200 h-3 rounded-full overflow-hidden mb-2">
//                 <div
//                   className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${getProgressColor(
//                     parseFloat(calculatePercentage(dailyNutrition.totalCarbs, userData.dailyCarbs))
//                   )}`}
//                   style={{
//                     width: `${Math.min(calculateProgress(dailyNutrition.totalCarbs, userData.dailyCarbs), 100)}%`,
//                   }}
//                 ></div>
//               </div>
              
//               <div className="flex items-center justify-between text-sm">
//                 <span className={`font-bold ${carbsStatus.color}`}>
//                   {calculatePercentage(dailyNutrition.totalCarbs, userData.dailyCarbs)}%
//                 </span>
//                 <span className="text-gray-600">
//                   {(userData.dailyCarbs - dailyNutrition.totalCarbs).toFixed(1)}g left
//                 </span>
//               </div>

//               {carbsStatus.status === 'exceeded' && (
//                 <div className="mt-3 bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-semibold">
//                   LIMIT EXCEEDED
//                 </div>
//               )}
//               {carbsStatus.status === 'warning' && (
//                 <div className="mt-3 bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-semibold">
//                   APPROACHING LIMIT
//                 </div>
//               )}
//             </div>

//             {/* Fat */}
//             <div className={`bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-2xl border-2 ${fatStatus.borderColor} relative overflow-hidden`}>
//               <div className="absolute top-2 right-2">
//                 <fatStatus.icon className={`w-6 h-6 ${fatStatus.color}`} />
//               </div>
//               <div className="flex items-center gap-3 mb-3">
//                 <div className="bg-blue-500 text-white p-2 rounded-lg">
//                   <Droplet className="w-5 h-5" />
//                 </div>
//                 <span className="font-bold text-gray-800">Fat</span>
//               </div>
//               <p className="text-3xl font-bold text-gray-900 mb-1">
//                 {dailyNutrition.totalFat.toFixed(0)}g
//               </p>
//               <p className="text-sm text-gray-600 mb-3">of {userData.dailyFat}g</p>
              
//               <div className="bg-gray-200 h-3 rounded-full overflow-hidden mb-2">
//                 <div
//                   className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${getProgressColor(
//                     parseFloat(calculatePercentage(dailyNutrition.totalFat, userData.dailyFat))
//                   )}`}
//                   style={{
//                     width: `${Math.min(calculateProgress(dailyNutrition.totalFat, userData.dailyFat), 100)}%`,
//                   }}
//                 ></div>
//               </div>
              
//               <div className="flex items-center justify-between text-sm">
//                 <span className={`font-bold ${fatStatus.color}`}>
//                   {calculatePercentage(dailyNutrition.totalFat, userData.dailyFat)}%
//                 </span>
//                 <span className="text-gray-600">
//                   {(userData.dailyFat - dailyNutrition.totalFat).toFixed(1)}g left
//                 </span>
//               </div>

//               {fatStatus.status === 'exceeded' && (
//                 <div className="mt-3 bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-semibold">
//                   LIMIT EXCEEDED
//                 </div>
//               )}
//               {fatStatus.status === 'warning' && (
//                 <div className="mt-3 bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-semibold">
//                   APPROACHING LIMIT
//                 </div>
//               )}
//             </div>
//           </div>

//           {/* Info Box */}
//           <div className="mt-6 bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-xl border border-blue-200 flex items-start gap-3">
//             <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
//             <div className="text-sm text-blue-800">
//               <p className="font-semibold mb-1">Understanding Your Progress:</p>
//               <ul className="space-y-1 text-blue-700">
//                 <li>• <span className="font-semibold">Green (0-50%):</span> Excellent progress, plenty of room left</li>
//                 <li>• <span className="font-semibold">Yellow (50-80%):</span> Good progress, be mindful</li>
//                 <li>• <span className="font-semibold">Orange (80-100%):</span> Approaching limit, choose carefully</li>
//                 <li>• <span className="font-semibold">Red (100%+):</span> Limit exceeded, avoid additional intake</li>
//               </ul>
//             </div>
//           </div>
//         </div>

//         {/* Today's Meals */}
//         <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-8">
//           <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
//             <Clock className="w-6 h-6 text-emerald-600" />
//             Today's Meals
//           </h2>

//           {foodLogs.length === 0 ? (
//             <div className="text-center py-12">
//               <div className="bg-gray-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
//                 <Drumstick className="w-10 h-10 text-gray-400" />
//               </div>
//               <p className="text-gray-500 font-medium mb-2">No meals logged today</p>
//               <p className="text-sm text-gray-400 mb-4">Start tracking your nutrition by logging your first meal</p>
//               <button
//                 onClick={() => router.push('/classifier')}
//                 className="bg-gradient-to-r from-emerald-500 to-green-600 text-white px-8 py-3 rounded-xl font-semibold hover:shadow-lg hover:scale-105 transition-all"
//               >
//                 Log Your First Meal
//               </button>
//             </div>
//           ) : (
//             <div className="space-y-4">
//               {foodLogs.map((log) => (
//                 <div
//                   key={log.id}
//                   className="bg-gradient-to-r from-emerald-50 to-green-50 p-5 rounded-xl border-2 border-emerald-200 hover:shadow-md transition-shadow"
//                 >
//                   <div className="flex items-start justify-between mb-3 flex-wrap gap-2">
//                     <div>
//                       <h3 className="text-lg font-bold text-gray-800">{log.foodName}</h3>
//                       <p className="text-sm text-gray-600 flex items-center gap-1">
//                         <Clock className="w-3 h-3" />
//                         {new Date(log.consumedAt).toLocaleTimeString('en-US', {
//                           hour: '2-digit',
//                           minute: '2-digit',
//                         })}
//                       </p>
//                     </div>
//                     <span
//                       className={`px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1 ${
//                         log.recommendation === 'recommended'
//                           ? 'bg-green-100 text-green-700'
//                           : log.recommendation === 'moderate'
//                           ? 'bg-yellow-100 text-yellow-700'
//                           : 'bg-red-100 text-red-700'
//                       }`}
//                     >
//                       {log.recommendation === 'recommended'
//                         ? '✓ Recommended'
//                         : log.recommendation === 'moderate'
//                         ? '⚠ Moderate'
//                         : '✗ Not Recommended'}
//                     </span>
//                   </div>
//                   <div className="grid grid-cols-4 gap-3">
//                     <div className="text-center bg-white p-3 rounded-lg border border-orange-200">
//                       <p className="text-xs text-gray-600 mb-1">Calories</p>
//                       <p className="font-bold text-gray-800 text-lg">{log.calories}</p>
//                     </div>
//                     <div className="text-center bg-white p-3 rounded-lg border border-purple-200">
//                       <p className="text-xs text-gray-600 mb-1">Protein</p>
//                       <p className="font-bold text-gray-800 text-lg">{log.protein}g</p>
//                     </div>
//                     <div className="text-center bg-white p-3 rounded-lg border border-amber-200">
//                       <p className="text-xs text-gray-600 mb-1">Carbs</p>
//                       <p className="font-bold text-gray-800 text-lg">{log.carbs}g</p>
//                     </div>
//                     <div className="text-center bg-white p-3 rounded-lg border border-blue-200">
//                       <p className="text-xs text-gray-600 mb-1">Fat</p>
//                       <p className="font-bold text-gray-800 text-lg">{log.fat}g</p>
//                     </div>
//                   </div>
//                 </div>
//               ))}
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }



















// src/app/(routes)/profile/page.tsx
'use client';
import { Suspense } from 'react';
import { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  User, Activity, TrendingUp, Calendar, Flame, Drumstick, Wheat, Droplet,
  Clock, CheckCircle, AlertCircle, Loader2, AlertTriangle, Trophy, Target,
  TrendingDown, Info
} from 'lucide-react';

interface UserData {
  name: string;
  email: string;
  age: number;
  height: number;
  weight: number;
  gender: string;
  bmi: number;
  medicalCondition: string | null;
  dailyCalories: number;
  dailyProtein: number;
  dailyCarbs: number;
  dailyFat: number;
}

interface DailyNutrition {
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  mealsCount: number;
}

interface FoodLog {
  id: number;
  foodName: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  recommendation: string;
  consumedAt: string;
}


function ProfileClient() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const searchParams = useSearchParams();
  const showSuccess = searchParams.get('logged') === 'true';

  const [userData, setUserData] = useState<UserData | null>(null);
  const [dailyNutrition, setDailyNutrition] = useState<DailyNutrition>({
    totalCalories: 0,
    totalProtein: 0,
    totalCarbs: 0,
    totalFat: 0,
    mealsCount: 0,
  });
  const [foodLogs, setFoodLogs] = useState<FoodLog[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/signin');
      return;
    }

    if (session?.user?.email) {
      fetchUserData();
      fetchDailyNutrition();
    }
  }, [status, session]);

  const fetchUserData = async () => {
    try {
      const response = await fetch('/api/user/profile');
      const data = await response.json();
      if (response.ok) {
        setUserData(data.user);
      }
    } catch (error) {
      console.error('Error fetching user data:', error);
    }
  };

  const fetchDailyNutrition = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/food/log');
      const data = await response.json();
      
      if (response.ok) {
        setDailyNutrition(data.dailyNutrition);
        setFoodLogs(data.foodLogs);
      }
    } catch (error) {
      console.error('Error fetching daily nutrition:', error);
    } finally {
      setLoading(false);
    }
  };

  if (status === 'loading' || loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-emerald-600 animate-spin mx-auto mb-4" />
          <p className="text-emerald-700 font-semibold">Loading profile...</p>
        </div>
      </div>
    );
  }

  if (!userData) return null;

  const calculateProgress = (current: number, target: number) => {
    return Math.min((current / target) * 100, 100);
  };

  const calculatePercentage = (current: number, target: number) => {
    return ((current / target) * 100).toFixed(1);
  };

  const getProgressColor = (percentage: number) => {
    if (percentage < 50) return 'from-green-500 to-emerald-600';
    if (percentage < 80) return 'from-yellow-500 to-orange-500';
    if (percentage < 100) return 'from-orange-500 to-red-500';
    return 'from-red-600 to-pink-600';
  };

  const getProgressStatus = (current: number, target: number) => {
    const percentage = (current / target) * 100;
    if (percentage >= 100) return { status: 'exceeded', icon: AlertTriangle, color: 'text-red-600', bgColor: 'bg-red-50', borderColor: 'border-red-200' };
    if (percentage >= 80) return { status: 'warning', icon: AlertCircle, color: 'text-orange-600', bgColor: 'bg-orange-50', borderColor: 'border-orange-200' };
    if (percentage >= 50) return { status: 'good', icon: TrendingUp, color: 'text-yellow-600', bgColor: 'bg-yellow-50', borderColor: 'border-yellow-200' };
    return { status: 'excellent', icon: CheckCircle, color: 'text-green-600', bgColor: 'bg-green-50', borderColor: 'border-green-200' };
  };

  const getBMICategory = (bmi: number) => {
    if (bmi < 18.5) return { text: 'Underweight', color: 'text-blue-600' };
    if (bmi < 25) return { text: 'Normal', color: 'text-green-600' };
    if (bmi < 30) return { text: 'Overweight', color: 'text-orange-600' };
    return { text: 'Obese', color: 'text-red-600' };
  };

  const bmiCategory = getBMICategory(userData.bmi);

  // Get overall status
  const caloriesStatus = getProgressStatus(dailyNutrition.totalCalories, userData.dailyCalories);
  const proteinStatus = getProgressStatus(dailyNutrition.totalProtein, userData.dailyProtein);
  const carbsStatus = getProgressStatus(dailyNutrition.totalCarbs, userData.dailyCarbs);
  const fatStatus = getProgressStatus(dailyNutrition.totalFat, userData.dailyFat);

  const hasExceeded = [caloriesStatus, proteinStatus, carbsStatus, fatStatus].some(s => s.status === 'exceeded');
  const hasWarning = [caloriesStatus, proteinStatus, carbsStatus, fatStatus].some(s => s.status === 'warning');

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 py-8 px-4">
      <div className="max-w-7xl mx-auto mt-12">
        {/* Success Message */}
        {showSuccess && (
          <div className="mb-6 bg-green-50 border border-green-200 text-green-800 px-6 py-4 rounded-xl flex items-center gap-3 animate-pulse">
            <CheckCircle className="w-6 h-6 flex-shrink-0" />
            <p className="font-semibold">Food logged successfully! Your daily nutrition has been updated.</p>
          </div>
        )}

        {/* Alert Messages */}
        {hasExceeded && (
          <div className="mb-6 bg-red-50 border-2 border-red-300 text-red-800 px-6 py-4 rounded-xl flex items-start gap-3">
            <AlertTriangle className="w-6 h-6 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-bold text-lg">⚠️ Daily Limit Exceeded!</p>
              <p className="text-sm mt-1">You've exceeded your daily limit for one or more nutrients. Consider lighter meals for the rest of the day.</p>
            </div>
          </div>
        )}

        {!hasExceeded && hasWarning && (
          <div className="mb-6 bg-orange-50 border-2 border-orange-300 text-orange-800 px-6 py-4 rounded-xl flex items-start gap-3">
            <AlertCircle className="w-6 h-6 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-bold text-lg">⚠️ Approaching Daily Limit</p>
              <p className="text-sm mt-1">You're getting close to your daily limits. Be mindful of your next meals.</p>
            </div>
          </div>
        )}

        {/* Header */}
        <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-8 mb-6">
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              <div className="bg-gradient-to-br from-emerald-500 to-green-600 text-white w-20 h-20 rounded-2xl flex items-center justify-center">
                <User className="w-10 h-10" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-800">{userData.name}</h1>
                <p className="text-gray-600">{userData.email}</p>
                <div className="flex items-center gap-4 mt-2 flex-wrap">
                  <span className="text-sm bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full font-semibold">
                    {userData.age} years old
                  </span>
                  <span className="text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-full font-semibold">
                    {userData.gender}
                  </span>
                </div>
              </div>
            </div>
            
            {/* Daily Progress Summary */}
            <div className="bg-gradient-to-br from-emerald-100 to-green-100 px-6 py-4 rounded-2xl border-2 border-emerald-300">
              <div className="flex items-center gap-2 mb-2">
                <Trophy className="w-5 h-5 text-emerald-700" />
                <span className="font-bold text-emerald-900">Today's Status</span>
              </div>
              <p className="text-2xl font-bold text-emerald-800">{dailyNutrition.mealsCount} meals logged</p>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6 mb-6">
          {/* Physical Stats */}
          <div className="bg-white rounded-2xl shadow-xl border border-emerald-100 p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-600" />
              Physical Stats
            </h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600 font-medium">Height</span>
                <span className="font-bold text-gray-800">{userData.height} cm</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600 font-medium">Weight</span>
                <span className="font-bold text-gray-800">{userData.weight} kg</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600 font-medium">BMI</span>
                <span className={`font-bold ${bmiCategory.color}`}>
                  {userData.bmi.toFixed(1)} ({bmiCategory.text})
                </span>
              </div>
            </div>
          </div>

          {/* Daily Goals */}
          <div className="bg-white rounded-2xl shadow-xl border border-emerald-100 p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-emerald-600" />
              Daily Goals
            </h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-gradient-to-r from-orange-50 to-orange-100 rounded-lg border border-orange-200">
                <span className="text-gray-700 font-medium">Calories</span>
                <span className="font-bold text-gray-800">{userData.dailyCalories} kcal</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg border border-purple-200">
                <span className="text-gray-700 font-medium">Protein</span>
                <span className="font-bold text-gray-800">{userData.dailyProtein}g</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gradient-to-r from-amber-50 to-amber-100 rounded-lg border border-amber-200">
                <span className="text-gray-700 font-medium">Carbs</span>
                <span className="font-bold text-gray-800">{userData.dailyCarbs}g</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg border border-blue-200">
                <span className="text-gray-700 font-medium">Fat</span>
                <span className="font-bold text-gray-800">{userData.dailyFat}g</span>
              </div>
            </div>
          </div>

          {/* Medical Info */}
          <div className="bg-white rounded-2xl shadow-xl border border-emerald-100 p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-emerald-600" />
              Medical Info
            </h2>
            <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-xl border border-blue-200 min-h-[120px] flex items-center">
              {userData.medicalCondition ? (
                <p className="text-gray-700 font-medium">{userData.medicalCondition}</p>
              ) : (
                <p className="text-gray-500 italic">No medical conditions reported</p>
              )}
            </div>
          </div>
        </div>

        {/* Today's Progress */}
        <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-8 mb-6">
          <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <Calendar className="w-6 h-6 text-emerald-600" />
              Today's Progress
            </h2>
            <div className="flex items-center gap-3">
              <div className="bg-emerald-100 text-emerald-700 px-4 py-2 rounded-full font-semibold">
                {dailyNutrition.mealsCount} meals logged
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Calories */}
            <div className={`bg-gradient-to-br from-orange-50 to-orange-100 p-6 rounded-2xl border-2 ${caloriesStatus.borderColor} relative overflow-hidden`}>
              <div className="absolute top-2 right-2">
                <caloriesStatus.icon className={`w-6 h-6 ${caloriesStatus.color}`} />
              </div>
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-orange-500 text-white p-2 rounded-lg">
                  <Flame className="w-5 h-5" />
                </div>
                <span className="font-bold text-gray-800">Calories</span>
              </div>
              <p className="text-3xl font-bold text-gray-900 mb-1">
                {dailyNutrition.totalCalories.toFixed(0)}
              </p>
              <p className="text-sm text-gray-600 mb-3">of {userData.dailyCalories} kcal</p>
              
              <div className="bg-gray-200 h-3 rounded-full overflow-hidden mb-2">
                <div
                  className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${getProgressColor(
                    parseFloat(calculatePercentage(dailyNutrition.totalCalories, userData.dailyCalories))
                  )}`}
                  style={{
                    width: `${Math.min(calculateProgress(dailyNutrition.totalCalories, userData.dailyCalories), 100)}%`,
                  }}
                ></div>
              </div>
              
              <div className="flex items-center justify-between text-sm">
                <span className={`font-bold ${caloriesStatus.color}`}>
                  {calculatePercentage(dailyNutrition.totalCalories, userData.dailyCalories)}%
                </span>
                <span className="text-gray-600">
                  {(userData.dailyCalories - dailyNutrition.totalCalories).toFixed(0)} left
                </span>
              </div>

              {caloriesStatus.status === 'exceeded' && (
                <div className="mt-3 bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-semibold">
                  LIMIT EXCEEDED
                </div>
              )}
              {caloriesStatus.status === 'warning' && (
                <div className="mt-3 bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-semibold">
                  APPROACHING LIMIT
                </div>
              )}
            </div>

            {/* Protein */}
            <div className={`bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-2xl border-2 ${proteinStatus.borderColor} relative overflow-hidden`}>
              <div className="absolute top-2 right-2">
                <proteinStatus.icon className={`w-6 h-6 ${proteinStatus.color}`} />
              </div>
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-purple-500 text-white p-2 rounded-lg">
                  <Drumstick className="w-5 h-5" />
                </div>
                <span className="font-bold text-gray-800">Protein</span>
              </div>
              <p className="text-3xl font-bold text-gray-900 mb-1">
                {dailyNutrition.totalProtein.toFixed(0)}g
              </p>
              <p className="text-sm text-gray-600 mb-3">of {userData.dailyProtein}g</p>
              
              <div className="bg-gray-200 h-3 rounded-full overflow-hidden mb-2">
                <div
                  className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${getProgressColor(
                    parseFloat(calculatePercentage(dailyNutrition.totalProtein, userData.dailyProtein))
                  )}`}
                  style={{
                    width: `${Math.min(calculateProgress(dailyNutrition.totalProtein, userData.dailyProtein), 100)}%`,
                  }}
                ></div>
              </div>
              
              <div className="flex items-center justify-between text-sm">
                <span className={`font-bold ${proteinStatus.color}`}>
                  {calculatePercentage(dailyNutrition.totalProtein, userData.dailyProtein)}%
                </span>
                <span className="text-gray-600">
                  {(userData.dailyProtein - dailyNutrition.totalProtein).toFixed(1)}g left
                </span>
              </div>

              {proteinStatus.status === 'exceeded' && (
                <div className="mt-3 bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-semibold">
                  LIMIT EXCEEDED
                </div>
              )}
              {proteinStatus.status === 'warning' && (
                <div className="mt-3 bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-semibold">
                  APPROACHING LIMIT
                </div>
              )}
            </div>

            {/* Carbs */}
            <div className={`bg-gradient-to-br from-amber-50 to-amber-100 p-6 rounded-2xl border-2 ${carbsStatus.borderColor} relative overflow-hidden`}>
              <div className="absolute top-2 right-2">
                <carbsStatus.icon className={`w-6 h-6 ${carbsStatus.color}`} />
              </div>
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-amber-500 text-white p-2 rounded-lg">
                  <Wheat className="w-5 h-5" />
                </div>
                <span className="font-bold text-gray-800">Carbs</span>
              </div>
              <p className="text-3xl font-bold text-gray-900 mb-1">
                {dailyNutrition.totalCarbs.toFixed(0)}g
              </p>
              <p className="text-sm text-gray-600 mb-3">of {userData.dailyCarbs}g</p>
              
              <div className="bg-gray-200 h-3 rounded-full overflow-hidden mb-2">
                <div
                  className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${getProgressColor(
                    parseFloat(calculatePercentage(dailyNutrition.totalCarbs, userData.dailyCarbs))
                  )}`}
                  style={{
                    width: `${Math.min(calculateProgress(dailyNutrition.totalCarbs, userData.dailyCarbs), 100)}%`,
                  }}
                ></div>
              </div>
              
              <div className="flex items-center justify-between text-sm">
                <span className={`font-bold ${carbsStatus.color}`}>
                  {calculatePercentage(dailyNutrition.totalCarbs, userData.dailyCarbs)}%
                </span>
                <span className="text-gray-600">
                  {(userData.dailyCarbs - dailyNutrition.totalCarbs).toFixed(1)}g left
                </span>
              </div>

              {carbsStatus.status === 'exceeded' && (
                <div className="mt-3 bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-semibold">
                  LIMIT EXCEEDED
                </div>
              )}
              {carbsStatus.status === 'warning' && (
                <div className="mt-3 bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-semibold">
                  APPROACHING LIMIT
                </div>
              )}
            </div>

            {/* Fat */}
            <div className={`bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-2xl border-2 ${fatStatus.borderColor} relative overflow-hidden`}>
              <div className="absolute top-2 right-2">
                <fatStatus.icon className={`w-6 h-6 ${fatStatus.color}`} />
              </div>
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-blue-500 text-white p-2 rounded-lg">
                  <Droplet className="w-5 h-5" />
                </div>
                <span className="font-bold text-gray-800">Fat</span>
              </div>
              <p className="text-3xl font-bold text-gray-900 mb-1">
                {dailyNutrition.totalFat.toFixed(0)}g
              </p>
              <p className="text-sm text-gray-600 mb-3">of {userData.dailyFat}g</p>
              
              <div className="bg-gray-200 h-3 rounded-full overflow-hidden mb-2">
                <div
                  className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${getProgressColor(
                    parseFloat(calculatePercentage(dailyNutrition.totalFat, userData.dailyFat))
                  )}`}
                  style={{
                    width: `${Math.min(calculateProgress(dailyNutrition.totalFat, userData.dailyFat), 100)}%`,
                  }}
                ></div>
              </div>
              
              <div className="flex items-center justify-between text-sm">
                <span className={`font-bold ${fatStatus.color}`}>
                  {calculatePercentage(dailyNutrition.totalFat, userData.dailyFat)}%
                </span>
                <span className="text-gray-600">
                  {(userData.dailyFat - dailyNutrition.totalFat).toFixed(1)}g left
                </span>
              </div>

              {fatStatus.status === 'exceeded' && (
                <div className="mt-3 bg-red-100 text-red-700 px-2 py-1 rounded text-xs font-semibold">
                  LIMIT EXCEEDED
                </div>
              )}
              {fatStatus.status === 'warning' && (
                <div className="mt-3 bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-semibold">
                  APPROACHING LIMIT
                </div>
              )}
            </div>
          </div>

          {/* Info Box */}
          <div className="mt-6 bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-xl border border-blue-200 flex items-start gap-3">
            <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-semibold mb-1">Understanding Your Progress:</p>
              <ul className="space-y-1 text-blue-700">
                <li>• <span className="font-semibold">Green (0-50%):</span> Excellent progress, plenty of room left</li>
                <li>• <span className="font-semibold">Yellow (50-80%):</span> Good progress, be mindful</li>
                <li>• <span className="font-semibold">Orange (80-100%):</span> Approaching limit, choose carefully</li>
                <li>• <span className="font-semibold">Red (100%+):</span> Limit exceeded, avoid additional intake</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Today's Meals */}
        <div className="bg-white rounded-3xl shadow-xl border border-emerald-100 p-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
            <Clock className="w-6 h-6 text-emerald-600" />
            Today's Meals
          </h2>

          {foodLogs.length === 0 ? (
            <div className="text-center py-12">
              <div className="bg-gray-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                <Drumstick className="w-10 h-10 text-gray-400" />
              </div>
              <p className="text-gray-500 font-medium mb-2">No meals logged today</p>
              <p className="text-sm text-gray-400 mb-4">Start tracking your nutrition by logging your first meal</p>
              <button
                onClick={() => router.push('/classifier')}
                className="bg-gradient-to-r from-emerald-500 to-green-600 text-white px-8 py-3 rounded-xl font-semibold hover:shadow-lg hover:scale-105 transition-all"
              >
                Log Your First Meal
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {foodLogs.map((log) => (
                <div
                  key={log.id}
                  className="bg-gradient-to-r from-emerald-50 to-green-50 p-5 rounded-xl border-2 border-emerald-200 hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-3 flex-wrap gap-2">
                    <div>
                      <h3 className="text-lg font-bold text-gray-800">{log.foodName}</h3>
                      <p className="text-sm text-gray-600 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(log.consumedAt).toLocaleTimeString('en-US', {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </p>
                    </div>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1 ${
                        log.recommendation === 'recommended'
                          ? 'bg-green-100 text-green-700'
                          : log.recommendation === 'moderate'
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {log.recommendation === 'recommended'
                        ? '✓ Recommended'
                        : log.recommendation === 'moderate'
                        ? '⚠ Moderate'
                        : '✗ Not Recommended'}
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-3">
                    <div className="text-center bg-white p-3 rounded-lg border border-orange-200">
                      <p className="text-xs text-gray-600 mb-1">Calories</p>
                      <p className="font-bold text-gray-800 text-lg">{log.calories}</p>
                    </div>
                    <div className="text-center bg-white p-3 rounded-lg border border-purple-200">
                      <p className="text-xs text-gray-600 mb-1">Protein</p>
                      <p className="font-bold text-gray-800 text-lg">{log.protein}g</p>
                    </div>
                    <div className="text-center bg-white p-3 rounded-lg border border-amber-200">
                      <p className="text-xs text-gray-600 mb-1">Carbs</p>
                      <p className="font-bold text-gray-800 text-lg">{log.carbs}g</p>
                    </div>
                    <div className="text-center bg-white p-3 rounded-lg border border-blue-200">
                      <p className="text-xs text-gray-600 mb-1">Fat</p>
                      <p className="font-bold text-gray-800 text-lg">{log.fat}g</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ProfilePage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="w-12 h-12 text-emerald-600 animate-spin mx-auto mb-4" />
            <p className="text-emerald-700 font-semibold">Loading profile...</p>
          </div>
        </div>
      }
    >
      <ProfileClient />
    </Suspense>
  );
}