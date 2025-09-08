from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import heapq
import os
import random
from datetime import datetime, date
from functools import lru_cache
import time
import statistics
from collections import defaultdict
import json

app = Flask(__name__)

# Macro focus weights
MACRO_FOCUS_MAP = {
	"protein": {"Protein": 2, "Carbs": 1, "Fat": 1, "Calories": 1},
	"carbs": {"Protein": 1, "Carbs": 2, "Fat": 1, "Calories": 1},
	"fat": {"Protein": 1, "Carbs": 1, "Fat": 2, "Calories": 1},
}

# Storage file for saved meal plans
SAVED_PLANS_FILE = "saved_meal_plans.json"

def load_saved_plans():
	"""Load saved meal plans from file"""
	try:
		if os.path.exists(SAVED_PLANS_FILE):
			with open(SAVED_PLANS_FILE, 'r') as f:
				return json.load(f)
		return {}
	except Exception as e:
		print(f"Error loading saved plans: {e}")
		return {}

def save_meal_plan(plan_date, plan_data, user_inputs, meal_split, macro_focus):
	"""Save a meal plan for a specific date"""
	try:
		saved_plans = load_saved_plans()
		
		plan_entry = {
			"date": plan_date,
			"plan": plan_data,
			"user_inputs": user_inputs,
			"meal_split": meal_split,
			"macro_focus": macro_focus,
			"saved_at": datetime.now().isoformat()
		}
		
		saved_plans[plan_date] = plan_entry
		
		with open(SAVED_PLANS_FILE, 'w') as f:
			json.dump(saved_plans, f, indent=2)
		
		return True
	except Exception as e:
		print(f"Error saving plan: {e}")
		return False

def get_saved_plan(plan_date):
	"""Get a saved meal plan for a specific date"""
	try:
		saved_plans = load_saved_plans()
		return saved_plans.get(plan_date)
	except Exception as e:
		print(f"Error loading plan: {e}")
		return None

def get_all_saved_dates():
	"""Get all dates with saved meal plans"""
	try:
		saved_plans = load_saved_plans()
		return sorted(saved_plans.keys(), reverse=True)  # Most recent first
	except Exception as e:
		print(f"Error loading dates: {e}")
		return []



def calculate_bmi(weight, height_cm):
	height_m = height_cm / 100
	return weight / (height_m ** 2)

def calculate_bmr(weight, height_cm, age, gender):
	if gender and gender.lower() == 'male':
		return 10 * weight + 6.25 * height_cm - 5 * age + 5
	return 10 * weight + 6.25 * height_cm - 5 * age - 161

def get_pal(activity):
	pal_map = {'sedentary': 1.53, 'moderate': 1.8, 'heavy': 2.3}
	return pal_map.get(activity, 1.53)

def adjust_calories(bmi, calories):
	if bmi < 18.5:
		return calories * 1.15
	elif bmi >= 25:
		return calories * 0.85
	return calories

def get_extra_protein(gender, pregnancy=None, lactating=None):
	if not gender or gender.lower() != 'female':
		return 0
	extra_map = {"2nd": 9.5, "3rd": 22, "0-6": 19.7, "7-12": 13.2}
	return extra_map.get(pregnancy, 0) + extra_map.get(lactating, 0)

def get_targets(user_inputs):
	bmi = calculate_bmi(user_inputs["weight"], user_inputs["height"])
	bmr = calculate_bmr(user_inputs["weight"], user_inputs["height"], user_inputs["age"], user_inputs["gender"])
	pal = get_pal(user_inputs["activity"])
	calculated_calories = adjust_calories(bmi, bmr * pal)
	
	# Use user-provided calories if available, otherwise use calculated
	if user_inputs.get("daily_calories") and user_inputs["daily_calories"] > 0:
		calories = float(user_inputs["daily_calories"])
	else:
		calories = calculated_calories
	
	extra_prot = get_extra_protein(user_inputs["gender"], user_inputs.get("pregnancy"), user_inputs.get("lactating"))
	protein_icmr = 0.83 * user_inputs["weight"] + extra_prot
	carbs = (0.60 * calories) / 4
	fats = (0.25 * calories) / 9
	protein = max((0.15 * calories) / 4, protein_icmr)
	return {
		"TDEE": round(calories),
		"Protein": round(protein),
		"Carbs": round(carbs),
		"Fat": round(fats),
		"BMI": round(bmi, 1),
		"Calculated_TDEE": round(calculated_calories),
	}



@lru_cache(maxsize=32)
def _read_excel_cached(file_path: str) -> pd.DataFrame:
	return pd.read_excel(file_path)

def load_dataset(file, region, preference):
	base_dir = os.path.dirname(os.path.abspath(__file__))
	file_path = os.path.join(base_dir, file)
	df = _read_excel_cached(file_path).copy()
	if region and region.lower() != "both":
		df = df[df["Region"].str.lower().str.strip() == region.lower()]
	if preference and preference.lower() == "veg":
		df = df[df["Veg/Non-Veg"].str.lower().str.strip() == "veg"]
	if df.empty:
		df = _read_excel_cached(file_path).copy()
	return df.reset_index(drop=True)

def filter_meal_candidates_by_calorie_window(df, meal_type, per_meal_calorie_target, window_pct=0.80):
	# Much wider calorie window to include more recipes
	lower = per_meal_calorie_target * (1 - window_pct)
	upper = per_meal_calorie_target * (1 + window_pct)
	
	if meal_type == "breakfast":
		calories = df["Main Qty"] * df["Main_Calories"] + df["Side Qty"] * df["Side_Calories"]
	else:
		calories = (df["Base Qty"] * df["Base_Calories"] + df["Subji Qty"] * df["Subji_Calories"] + df["Side Qty"] * df["Side_Calories"]) 
	
	# Use all recipes, just sort by calorie proximity
	df = df.assign(_cals=calories)
	df = df.iloc[(df["_cals"] - per_meal_calorie_target).abs().sort_values().index]
	filtered = df.drop(columns=["_cals"]).head(200).reset_index(drop=True)
	
	return filtered



def generate_meal_combinations(df, meal_type, target_calories=None):
	multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
	combos = []
	for idx, row in df.iterrows():
		if meal_type == "breakfast":
			for mul in multipliers:
				m_qty = float(row["Main Qty"]) * mul
				s_qty = float(row["Side Qty"]) * mul
				cals = m_qty * row["Main_Calories"] + s_qty * row["Side_Calories"]
				prot = m_qty * row["Main_Protein"] + s_qty * row["Side_Protein"]
				carb = m_qty * row["Main_Carbs"] + s_qty * row["Side_Carbs"]
				fat  = m_qty * row["Main_Fat"] + s_qty * row["Side_Fat"]
				label = f"{m_qty:.1f} {row['Main Unit']} {row['Main']} + {s_qty:.1f} {row['Side Unit']} {row['Side']}"
				
				# Calculate how close this meal is to target calories
				calorie_match = 1.0 - min(abs(cals - target_calories) / target_calories, 1.0) if target_calories else 0.0
				
				combos.append({
					"label": label,
					"bucket_id": f"{meal_type}_{idx}",
					"norm_id": f"{row['Main']}|{row['Side']}",
					"macros": {"Calories": cals, "Protein": prot, "Carbs": carb, "Fat": fat},
					"calorie_match": calorie_match
				})
		else:
			for mul in multipliers:
				b_qty = float(row["Base Qty"]) * mul
				j_qty = float(row["Subji Qty"]) * mul
				sd_qty = float(row["Side Qty"]) * mul
				cals = b_qty * row["Base_Calories"] + j_qty * row["Subji_Calories"] + sd_qty * row["Side_Calories"]
				prot = b_qty * row["Base_Protein"] + j_qty * row["Subji_Protein"] + sd_qty * row["Side_Protein"]
				carb = b_qty * row["Base_Carbs"] + j_qty * row["Subji_Carbs"] + sd_qty * row["Side_Carbs"]
				fat  = b_qty * row["Base_Fat"] + j_qty * row["Subji_Fat"] + sd_qty * row["Side_Fat"]
				label = f"{b_qty:.1f} {row['Base Unit']} {row['Base']} + {j_qty:.1f} {row['Subji Unit']} {row['Subji']} + {sd_qty:.1f} {row['Side Unit']} {row['Side']}"
				
				# Calculate how close this meal is to target calories
				calorie_match = 1.0 - min(abs(cals - target_calories) / target_calories, 1.0) if target_calories else 0.0
				
				combos.append({
					"label": label,
					"bucket_id": f"{meal_type}_{idx}",
					"norm_id": f"{row['Base']}|{row['Subji']}|{row['Side']}",
					"macros": {"Calories": cals, "Protein": prot, "Carbs": carb, "Fat": fat},
					"calorie_match": calorie_match
				})
	
	# Sort combinations by calorie match to prioritize better matches
	combos.sort(key=lambda x: x.get("calorie_match", 0), reverse=True)
	return combos



def score_meal_plan(plan, targets, meal_split, macro_priority):
	if not plan or not plan.get("total_macros") or not plan.get("meals"):
		return 0.0
	
	# FLEXIBLE meal split - allow variety while maintaining reasonable distribution
	meal_split_bonus = 0.0
	try:
		bfast_cals = plan["meals"]["breakfast"]["macros"]["Calories"] if plan["meals"].get("breakfast") else 0
		lunch_cals = plan["meals"]["lunch"]["macros"]["Calories"] if plan["meals"].get("lunch") else 0
		dinner_cals = plan["meals"]["dinner"]["macros"]["Calories"] if plan["meals"].get("dinner") else 0
		total_cals = bfast_cals + lunch_cals + dinner_cals
		
		if total_cals > 0:
			bfast_pct = (bfast_cals / total_cals) * 100
			lunch_pct = (lunch_cals / total_cals) * 100
			dinner_pct = (dinner_cals / total_cals) * 100
			
			# Give bonus for reasonable distribution, but don't reject
			bfast_diff = abs(bfast_pct - meal_split["breakfast"])
			lunch_diff = abs(lunch_pct - meal_split["lunch"])
			dinner_diff = abs(dinner_pct - meal_split["dinner"])
			
			# Small bonus for following preferences, but allow flexibility
			meal_split_bonus = max(0.0, 1.0 - (bfast_diff + lunch_diff + dinner_diff) / 60)
	except Exception:
		meal_split_bonus = 0.0
	
	calorie_target = targets["TDEE"]
	calorie_actual = plan["total_macros"].get("Calories", 0)
	calorie_deviation_pct = abs(calorie_actual - calorie_target) / calorie_target if calorie_target > 0 else 1.0
	
	if calorie_deviation_pct <= 0.05:
		calorie_score = 1.0
	elif calorie_deviation_pct <= 0.10:
		calorie_score = 0.5
	elif calorie_deviation_pct <= 0.15:
		calorie_score = 0.2
	else:
		calorie_score = 0.0
	
	total = 5.0 * calorie_score
	
	# Add flexible meal split bonus (small weight to allow variety)
	total += 1.0 * meal_split_bonus
	
	# Score other macros
	for macro in ["Protein", "Carbs", "Fat"]:
		tgt = targets[macro]
		act = plan["total_macros"].get(macro, 0)
		s = 0.0 if tgt <= 0 else max(0.0, 1.0 - abs(act - tgt) / tgt)
		total += macro_priority.get(macro, 1.0) * s * 0.5
	
	return total

def calculate_accuracy_metrics(plan, targets, meal_split):
	"""
	Calculate accuracy metrics for how close a meal plan is to target values
	"""
	if not plan or not plan.get("total_macros") or not plan.get("meals"):
		return {
			'calorie_accuracy': 0.0,
			'macro_accuracy': 0.0,
			'meal_distribution_accuracy': 0.0,
			'overall_accuracy': 0.0,
			'calorie_deviation_pct': 1.0,
			'macro_deviations': {'Protein': 1.0, 'Carbs': 1.0, 'Fat': 1.0},
			'meal_distribution_deviations': {'breakfast': 1.0, 'lunch': 1.0, 'dinner': 1.0}
		}
	
	# Calorie accuracy
	calorie_target = targets["TDEE"]
	calorie_actual = plan["total_macros"].get("Calories", 0)
	calorie_deviation_pct = abs(calorie_actual - calorie_target) / calorie_target if calorie_target > 0 else 1.0
	calorie_accuracy = max(0.0, 1.0 - calorie_deviation_pct)
	
	# Macro accuracy
	macro_deviations = {}
	macro_accuracies = []
	for macro in ["Protein", "Carbs", "Fat"]:
		tgt = targets[macro]
		act = plan["total_macros"].get(macro, 0)
		if tgt > 0:
			deviation = abs(act - tgt) / tgt
			macro_deviations[macro] = deviation
			macro_accuracies.append(max(0.0, 1.0 - deviation))
		else:
			macro_deviations[macro] = 1.0
			macro_accuracies.append(0.0)
	
	macro_accuracy = sum(macro_accuracies) / len(macro_accuracies) if macro_accuracies else 0.0
	
	# Meal distribution accuracy
	meal_distribution_deviations = {}
	meal_distribution_accuracies = []
	try:
		bfast_cals = plan["meals"]["breakfast"]["macros"]["Calories"] if plan["meals"].get("breakfast") else 0
		lunch_cals = plan["meals"]["lunch"]["macros"]["Calories"] if plan["meals"].get("lunch") else 0
		dinner_cals = plan["meals"]["dinner"]["macros"]["Calories"] if plan["meals"].get("dinner") else 0
		total_cals = bfast_cals + lunch_cals + dinner_cals
		
		if total_cals > 0:
			bfast_pct = (bfast_cals / total_cals) * 100
			lunch_pct = (lunch_cals / total_cals) * 100
			dinner_pct = (dinner_cals / total_cals) * 100
			
			bfast_deviation = abs(bfast_pct - meal_split["breakfast"]) / 100
			lunch_deviation = abs(lunch_pct - meal_split["lunch"]) / 100
			dinner_deviation = abs(dinner_pct - meal_split["dinner"]) / 100
			
			meal_distribution_deviations = {
				'breakfast': bfast_deviation,
				'lunch': lunch_deviation,
				'dinner': dinner_deviation
			}
			
			meal_distribution_accuracies = [
				max(0.0, 1.0 - bfast_deviation),
				max(0.0, 1.0 - lunch_deviation),
				max(0.0, 1.0 - dinner_deviation)
			]
			
			meal_distribution_accuracy = sum(meal_distribution_accuracies) / len(meal_distribution_accuracies)
		else:
			meal_distribution_accuracy = 0.0
			meal_distribution_deviations = {'breakfast': 1.0, 'lunch': 1.0, 'dinner': 1.0}
	except Exception:
		meal_distribution_accuracy = 0.0
		meal_distribution_deviations = {'breakfast': 1.0, 'lunch': 1.0, 'dinner': 1.0}
	
	# Overall accuracy (weighted average)
	overall_accuracy = (calorie_accuracy * 0.4 + macro_accuracy * 0.4 + meal_distribution_accuracy * 0.2)
	
	return {
		'calorie_accuracy': calorie_accuracy,
		'macro_accuracy': macro_accuracy,
		'meal_distribution_accuracy': meal_distribution_accuracy,
		'overall_accuracy': overall_accuracy,
		'calorie_deviation_pct': calorie_deviation_pct,
		'macro_deviations': macro_deviations,
		'meal_distribution_deviations': meal_distribution_deviations
	}

def generate_brute_force_plans(bfast_combos, lunch_combos, dinner_combos, targets, meal_split, macro_priority, max_combinations=10000):
	"""
	Brute Force Algorithm: Systematically evaluate combinations with scoring
	- Pros: Guaranteed to find optimal solution within search space
	- Cons: Exponential time complexity, limited by max_combinations
	"""
	start_time = time.time()
	combinations_evaluated = 0
	best_plans = []
	
	# Limit search space to prevent excessive computation
	total_possible = len(bfast_combos) * len(lunch_combos) * len(dinner_combos)
	if total_possible > max_combinations:
		# Sample combinations systematically
		bfast_sample = bfast_combos[:min(len(bfast_combos), int(max_combinations ** (1/3)))]
		lunch_sample = lunch_combos[:min(len(lunch_combos), int(max_combinations ** (1/3)))]
		dinner_sample = dinner_combos[:min(len(dinner_combos), int(max_combinations ** (1/3)))]
	else:
		bfast_sample, lunch_sample, dinner_sample = bfast_combos, lunch_combos, dinner_combos
	
	for b in bfast_sample:
		for l in lunch_sample:
			for d in dinner_sample:
				combinations_evaluated += 1
				
				total_macros = {
					"Calories": b["macros"]["Calories"] + l["macros"]["Calories"] + d["macros"]["Calories"],
					"Protein": b["macros"]["Protein"] + l["macros"]["Protein"] + d["macros"]["Protein"],
					"Carbs": b["macros"]["Carbs"] + l["macros"]["Carbs"] + d["macros"]["Carbs"],
					"Fat": b["macros"]["Fat"] + l["macros"]["Fat"] + d["macros"]["Fat"],
				}
				
				plan = {"meals": {"breakfast": b, "lunch": l, "dinner": d}, "total_macros": total_macros}
				score = score_meal_plan(plan, targets, meal_split, macro_priority)
				
				best_plans.append((score, plan))
				best_plans.sort(key=lambda x: x[0], reverse=True)
				best_plans = best_plans[:20]  # Keep top 20
				
				if combinations_evaluated >= max_combinations:
					break
			if combinations_evaluated >= max_combinations:
				break
		if combinations_evaluated >= max_combinations:
			break
	
	execution_time = time.time() - start_time
	
	# Calculate accuracy metrics for best plan
	best_plan = best_plans[0][1] if best_plans else None
	accuracy_metrics = calculate_accuracy_metrics(best_plan, targets, meal_split) if best_plan else {}
	
	metrics = {
		'algorithm': 'brute_force',
		'execution_time': execution_time,
		'combinations_evaluated': combinations_evaluated,
		'total_possible': total_possible,
		'search_space_coverage': combinations_evaluated / total_possible if total_possible > 0 else 0,
		'best_score': best_plans[0][0] if best_plans else 0,
		'worst_score': best_plans[-1][0] if best_plans else 0,
		'avg_score': sum(score for score, _ in best_plans) / len(best_plans) if best_plans else 0,
		'accuracy_metrics': accuracy_metrics
	}
	
	return [plan for score, plan in best_plans], metrics


def generate_genetic_algorithm_plans_with_metrics(bfast_combos, lunch_combos, dinner_combos, targets, meal_split, macro_priority, population_size=100, generations=50, top_n=100):
	"""
	Simplified Genetic Algorithm - Removed unnecessary complexity while maintaining quality
	"""
	start_time = time.time()
	metrics = {
		'algorithm': 'genetic',
		'execution_time': 0,
		'fitness_evaluations': 0,
		'population_size': population_size,
		'generations': generations
	}
	
	# Initialize population with random meal combinations
	population = []
	for _ in range(population_size):
		b = random.choice(bfast_combos)
		l = random.choice(lunch_combos)
		d = random.choice(dinner_combos)
		
		total_macros = {
			"Calories": b["macros"]["Calories"] + l["macros"]["Calories"] + d["macros"]["Calories"],
			"Protein": b["macros"]["Protein"] + l["macros"]["Protein"] + d["macros"]["Protein"],
			"Carbs": b["macros"]["Carbs"] + l["macros"]["Carbs"] + d["macros"]["Carbs"],
			"Fat": b["macros"]["Fat"] + l["macros"]["Fat"] + d["macros"]["Fat"],
		}
		
		plan = {"meals": {"breakfast": b, "lunch": l, "dinner": d}, "total_macros": total_macros}
		population.append(plan)
	
	# Evolution loop
	for generation in range(generations):
		# Score all plans in current population
		scored_population = [(score_meal_plan(plan, targets, meal_split, macro_priority), plan) for plan in population]
		scored_population.sort(key=lambda x: x[0], reverse=True)
		
		metrics['fitness_evaluations'] += len(population)
		
		# Simplified Selection: Keep top 20 (elite) + take top 70 for breeding
		elite = [plan for score, plan in scored_population[:20]]  # Top 20
		breeders = [plan for score, plan in scored_population[:70]]  # Top 70 for breeding
		
		# Create new population
		new_population = elite.copy()  # Keep top 20
		
		# Simplified Crossover: Create 80 new plans by mixing breeders
		while len(new_population) < population_size:
			parent1 = random.choice(breeders)
			parent2 = random.choice(breeders)
			
			# Simple crossover: mix meals from both parents
			child = {
				"meals": {
					"breakfast": parent1["meals"]["breakfast"],
					"lunch": parent2["meals"]["lunch"],
					"dinner": parent1["meals"]["dinner"]
				}
			}
			
			# Recalculate total macros
			child["total_macros"] = {
				"Calories": child["meals"]["breakfast"]["macros"]["Calories"] + 
						   child["meals"]["lunch"]["macros"]["Calories"] + 
						   child["meals"]["dinner"]["macros"]["Calories"],
				"Protein": child["meals"]["breakfast"]["macros"]["Protein"] + 
						  child["meals"]["lunch"]["macros"]["Protein"] + 
						  child["meals"]["dinner"]["macros"]["Protein"],
				"Carbs": child["meals"]["breakfast"]["macros"]["Carbs"] + 
						child["meals"]["lunch"]["macros"]["Carbs"] + 
						child["meals"]["dinner"]["macros"]["Carbs"],
				"Fat": child["meals"]["breakfast"]["macros"]["Fat"] + 
					   child["meals"]["lunch"]["macros"]["Fat"] + 
					   child["meals"]["dinner"]["macros"]["Fat"],
			}
			
			new_population.append(child)
		
		# Simplified Mutation: Randomly change 5% of non-elite plans
		for i in range(20, len(new_population)):
			if random.random() < 0.05:  # 5% mutation rate
				meal_type = random.choice(["breakfast", "lunch", "dinner"])
				if meal_type == "breakfast":
					new_population[i]["meals"]["breakfast"] = random.choice(bfast_combos)
				elif meal_type == "lunch":
					new_population[i]["meals"]["lunch"] = random.choice(lunch_combos)
				else:
					new_population[i]["meals"]["dinner"] = random.choice(dinner_combos)
				
				# Recalculate macros after mutation
				new_population[i]["total_macros"] = {
					"Calories": new_population[i]["meals"]["breakfast"]["macros"]["Calories"] + 
							   new_population[i]["meals"]["lunch"]["macros"]["Calories"] + 
							   new_population[i]["meals"]["dinner"]["macros"]["Calories"],
					"Protein": new_population[i]["meals"]["breakfast"]["macros"]["Protein"] + 
							  new_population[i]["meals"]["lunch"]["macros"]["Protein"] + 
							  new_population[i]["meals"]["dinner"]["macros"]["Protein"],
					"Carbs": new_population[i]["meals"]["breakfast"]["macros"]["Carbs"] + 
							new_population[i]["meals"]["lunch"]["macros"]["Carbs"] + 
							new_population[i]["meals"]["dinner"]["macros"]["Carbs"],
					"Fat": new_population[i]["meals"]["breakfast"]["macros"]["Fat"] + 
						   new_population[i]["meals"]["lunch"]["macros"]["Fat"] + 
						   new_population[i]["meals"]["dinner"]["macros"]["Fat"],
				}
		
		population = new_population
	
	# Return top N plans from final population
	final_scored = [(score_meal_plan(plan, targets, meal_split, macro_priority), plan) for plan in population]
	final_scored.sort(key=lambda x: x[0], reverse=True)
	
	metrics['execution_time'] = time.time() - start_time
	metrics['best_score'] = final_scored[0][0] if final_scored else 0
	metrics['worst_score'] = final_scored[-1][0] if final_scored else 0
	metrics['avg_score'] = sum(score for score, _ in final_scored) / len(final_scored) if final_scored else 0
	
	# Calculate accuracy metrics for best plan
	best_plan = final_scored[0][1] if final_scored else None
	accuracy_metrics = calculate_accuracy_metrics(best_plan, targets, meal_split) if best_plan else {}
	metrics['accuracy_metrics'] = accuracy_metrics
	
	return [plan for score, plan in final_scored[:top_n]], metrics


def compare_algorithms(bfast_combos, lunch_combos, dinner_combos, targets, meal_split, macro_priority):
	"""
	Compare Brute Force and Genetic algorithms
	"""
	print("Running algorithm comparison...")
	
	# Run both algorithms
	brute_plans, brute_metrics = generate_brute_force_plans(bfast_combos, lunch_combos, dinner_combos, targets, meal_split, macro_priority)
	genetic_plans, genetic_metrics = generate_genetic_algorithm_plans_with_metrics(bfast_combos, lunch_combos, dinner_combos, targets, meal_split, macro_priority)
	
	# Compile comparison results
	comparison = {
		'brute_force': brute_metrics,
		'genetic': genetic_metrics,
		'summary': {
			'fastest': min([(brute_metrics['execution_time'], 'brute_force'), 
						   (genetic_metrics['execution_time'], 'genetic')], key=lambda x: x[0])[1],
			'best_score': max([(brute_metrics['best_score'], 'brute_force'), 
							  (genetic_metrics['best_score'], 'genetic')], key=lambda x: x[0])[1],
			'most_efficient': min([(brute_metrics['execution_time'] / max(brute_metrics['best_score'], 0.1), 'brute_force'), 
								  (genetic_metrics['execution_time'] / max(genetic_metrics['best_score'], 0.1), 'genetic')], key=lambda x: x[0])[1],
			'most_accurate': max([(brute_metrics.get('accuracy_metrics', {}).get('overall_accuracy', 0), 'brute_force'), 
								 (genetic_metrics.get('accuracy_metrics', {}).get('overall_accuracy', 0), 'genetic')], key=lambda x: x[0])[1],
			'closest_to_targets': max([(1.0 - brute_metrics.get('accuracy_metrics', {}).get('calorie_deviation_pct', 1.0), 'brute_force'), 
									  (1.0 - genetic_metrics.get('accuracy_metrics', {}).get('calorie_deviation_pct', 1.0), 'genetic')], key=lambda x: x[0])[1]
		}
	}
	
	# Print comparison summary
	print(f"\n=== ALGORITHM COMPARISON RESULTS ===")
	print(f"Fastest: {comparison['summary']['fastest']} algorithm")
	print(f"Best Score: {comparison['summary']['best_score']} algorithm")
	print(f"Most Efficient (time/score): {comparison['summary']['most_efficient']} algorithm")
	print(f"Most Accurate Overall: {comparison['summary']['most_accurate']} algorithm")
	print(f"Closest to Target Values: {comparison['summary']['closest_to_targets']} algorithm")
	print(f"\nDetailed Metrics:")
	print(f"Brute Force: {brute_metrics['execution_time']:.3f}s, Score: {brute_metrics['best_score']:.2f}, Accuracy: {brute_metrics.get('accuracy_metrics', {}).get('overall_accuracy', 0):.3f}")
	print(f"Genetic: {genetic_metrics['execution_time']:.3f}s, Score: {genetic_metrics['best_score']:.2f}, Accuracy: {genetic_metrics.get('accuracy_metrics', {}).get('overall_accuracy', 0):.3f}")
	
	# Print accuracy breakdown
	print(f"\n=== ACCURACY BREAKDOWN ===")
	for algo_name, algo_metrics in [('Brute Force', brute_metrics), ('Genetic', genetic_metrics)]:
		acc_metrics = algo_metrics.get('accuracy_metrics', {})
		print(f"\n{algo_name}:")
		print(f"  Calorie Accuracy: {acc_metrics.get('calorie_accuracy', 0):.3f} (Deviation: {acc_metrics.get('calorie_deviation_pct', 1.0)*100:.1f}%)")
		print(f"  Macro Accuracy: {acc_metrics.get('macro_accuracy', 0):.3f}")
		print(f"  Meal Distribution Accuracy: {acc_metrics.get('meal_distribution_accuracy', 0):.3f}")
		print(f"  Overall Accuracy: {acc_metrics.get('overall_accuracy', 0):.3f}")
	
	# Add best plans to comparison
	comparison['brute_force']['best_plan'] = brute_plans[0] if brute_plans else None
	comparison['genetic']['best_plan'] = genetic_plans[0] if genetic_plans else None
	
	return comparison

def generate_genetic_algorithm_plans(bfast_combos, lunch_combos, dinner_combos, targets, meal_split, macro_priority, population_size=100, generations=50, top_n=100):
	"""
	Simplified Genetic Algorithm for meal planning:
	- Population: Collection of meal plans
	- Selection: Keep best 20 + breed from top 70
	- Crossover: Simple mixing of meals from parents
	- Mutation: 5% random changes for variety
	"""
	
	# Initialize population with random meal combinations
	population = []
	for _ in range(population_size):
		b = random.choice(bfast_combos)
		l = random.choice(lunch_combos)
		d = random.choice(dinner_combos)
		
		total_macros = {
			"Calories": b["macros"]["Calories"] + l["macros"]["Calories"] + d["macros"]["Calories"],
			"Protein": b["macros"]["Protein"] + l["macros"]["Protein"] + d["macros"]["Protein"],
			"Carbs": b["macros"]["Carbs"] + l["macros"]["Carbs"] + d["macros"]["Carbs"],
			"Fat": b["macros"]["Fat"] + l["macros"]["Fat"] + d["macros"]["Fat"],
		}
		
		plan = {"meals": {"breakfast": b, "lunch": l, "dinner": d}, "total_macros": total_macros}
		population.append(plan)
	
	# Evolution loop
	for generation in range(generations):
		# Score all plans in current population
		scored_population = [(score_meal_plan(plan, targets, meal_split, macro_priority), plan) for plan in population]
		scored_population.sort(key=lambda x: x[0], reverse=True)
		
		# Simplified Selection: Keep top 20 (elite) + take top 70 for breeding
		elite = [plan for score, plan in scored_population[:20]]  # Top 20
		breeders = [plan for score, plan in scored_population[:70]]  # Top 70 for breeding
		
		# Create new population
		new_population = elite.copy()  # Keep top 20
		
		# Simplified Crossover: Create 80 new plans by mixing breeders
		while len(new_population) < population_size:
			parent1 = random.choice(breeders)
			parent2 = random.choice(breeders)
			
			# Simple crossover: mix meals from both parents
			child = {
				"meals": {
					"breakfast": parent1["meals"]["breakfast"],
					"lunch": parent2["meals"]["lunch"],
					"dinner": parent1["meals"]["dinner"]
				}
			}
			
			# Recalculate total macros
			child["total_macros"] = {
				"Calories": child["meals"]["breakfast"]["macros"]["Calories"] + 
						   child["meals"]["lunch"]["macros"]["Calories"] + 
						   child["meals"]["dinner"]["macros"]["Calories"],
				"Protein": child["meals"]["breakfast"]["macros"]["Protein"] + 
						  child["meals"]["lunch"]["macros"]["Protein"] + 
						  child["meals"]["dinner"]["macros"]["Protein"],
				"Carbs": child["meals"]["breakfast"]["macros"]["Carbs"] + 
						child["meals"]["lunch"]["macros"]["Carbs"] + 
						child["meals"]["dinner"]["macros"]["Carbs"],
				"Fat": child["meals"]["breakfast"]["macros"]["Fat"] + 
					   child["meals"]["lunch"]["macros"]["Fat"] + 
					   child["meals"]["dinner"]["macros"]["Fat"],
			}
			
			new_population.append(child)
		
		# Simplified Mutation: Randomly change 5% of non-elite plans
		for i in range(20, len(new_population)):
			if random.random() < 0.05:  # 5% mutation rate
				meal_type = random.choice(["breakfast", "lunch", "dinner"])
				if meal_type == "breakfast":
					new_population[i]["meals"]["breakfast"] = random.choice(bfast_combos)
				elif meal_type == "lunch":
					new_population[i]["meals"]["lunch"] = random.choice(lunch_combos)
				else:
					new_population[i]["meals"]["dinner"] = random.choice(dinner_combos)
				
				# Recalculate macros after mutation
				new_population[i]["total_macros"] = {
					"Calories": new_population[i]["meals"]["breakfast"]["macros"]["Calories"] + 
							   new_population[i]["meals"]["lunch"]["macros"]["Calories"] + 
							   new_population[i]["meals"]["dinner"]["macros"]["Calories"],
					"Protein": new_population[i]["meals"]["breakfast"]["macros"]["Protein"] + 
							  new_population[i]["meals"]["lunch"]["macros"]["Protein"] + 
							  new_population[i]["meals"]["dinner"]["macros"]["Protein"],
					"Carbs": new_population[i]["meals"]["breakfast"]["macros"]["Carbs"] + 
							new_population[i]["meals"]["lunch"]["macros"]["Carbs"] + 
							new_population[i]["meals"]["dinner"]["macros"]["Carbs"],
					"Fat": new_population[i]["meals"]["breakfast"]["macros"]["Fat"] + 
						   new_population[i]["meals"]["lunch"]["macros"]["Fat"] + 
						   new_population[i]["meals"]["dinner"]["macros"]["Fat"],
				}
		
		population = new_population
	
	# Return top N plans from final population
	final_scored = [(score_meal_plan(plan, targets, meal_split, macro_priority), plan) for plan in population]
	final_scored.sort(key=lambda x: x[0], reverse=True)
	
	return [plan for score, plan in final_scored[:top_n]]

def generate_plans(user_inputs, meal_split, macro_priority, num_restarts=8):
	targets = get_targets(user_inputs)
	bfast_target = (targets["TDEE"] * meal_split["breakfast"]) / 100
	lunch_target = (targets["TDEE"] * meal_split["lunch"]) / 100
	dinner_target = (targets["TDEE"] * meal_split["dinner"]) / 100

	all_plans = []
	for r in range(num_restarts):
		b_df = load_dataset("Breakfast_Adjusted_Max.xlsx", user_inputs.get("region"), user_inputs.get("preference"))
		l_df = load_dataset("Lunch_Adjusted_Max.xlsx", user_inputs.get("region"), user_inputs.get("preference"))
		d_df = load_dataset("Dinner_Adjusted_Max.xlsx", user_inputs.get("region"), user_inputs.get("preference"))
		
		b_df = filter_meal_candidates_by_calorie_window(b_df, "breakfast", bfast_target, 0.80)
		l_df = filter_meal_candidates_by_calorie_window(l_df, "lunch", lunch_target, 0.80)
		d_df = filter_meal_candidates_by_calorie_window(d_df, "dinner", dinner_target, 0.80)
		
		b_combos = generate_meal_combinations(b_df, "breakfast", bfast_target)
		l_combos = generate_meal_combinations(l_df, "lunch", lunch_target)
		d_combos = generate_meal_combinations(d_df, "dinner", dinner_target)
		
		plans = generate_genetic_algorithm_plans(b_combos, l_combos, d_combos, targets, meal_split, macro_priority, population_size=100, generations=50, top_n=200)
		all_plans.extend(plans)

	# Sort by score but add some randomization for variety
	all_plans.sort(key=lambda p: score_meal_plan(p, targets, meal_split, macro_priority), reverse=True)
	
	# Shuffle top plans slightly to ensure variety
	import random
	top_plans = all_plans[:50]  # Top 50 plans
	random.shuffle(top_plans)
	all_plans = top_plans + all_plans[50:]  # Keep top shuffled, rest as is
	
	unique_plans = []
	seen_combinations = set()
	
	for p in all_plans:
		if not p or not p.get("meals"):
			continue
			
		bfast_meal = p["meals"].get("breakfast")
		lunch_meal = p["meals"].get("lunch")
		dinner_meal = p["meals"].get("dinner")
		
		if not bfast_meal or not lunch_meal or not dinner_meal:
			continue
		
		b_bucket = bfast_meal.get("bucket_id", "")
		l_bucket = lunch_meal.get("bucket_id", "")
		d_bucket = dinner_meal.get("bucket_id", "")
		
		if not b_bucket or not l_bucket or not d_bucket:
			continue
		
		combination_key = f"{b_bucket}|{l_bucket}|{d_bucket}"
		
		if combination_key in seen_combinations:
			continue
		
		unique_plans.append(p)
		seen_combinations.add(combination_key)
		
		if len(unique_plans) >= 20:
			break

	return targets, unique_plans[:20]

def generate_weekly_plans(user_inputs, meal_split, macro_priority, num_days=7):
	targets = get_targets(user_inputs)
	bfast_target = (targets["TDEE"] * meal_split["breakfast"]) / 100
	lunch_target = (targets["TDEE"] * meal_split["lunch"]) / 100
	dinner_target = (targets["TDEE"] * meal_split["dinner"]) / 100

	b_df = load_dataset("Breakfast_Adjusted_Max.xlsx", user_inputs.get("region"), user_inputs.get("preference"))
	l_df = load_dataset("Lunch_Adjusted_Max.xlsx", user_inputs.get("region"), user_inputs.get("preference"))
	d_df = load_dataset("Dinner_Adjusted_Max.xlsx", user_inputs.get("region"), user_inputs.get("preference"))
	
	b_df = filter_meal_candidates_by_calorie_window(b_df, "breakfast", bfast_target, 0.80)
	l_df = filter_meal_candidates_by_calorie_window(l_df, "lunch", lunch_target, 0.80)
	d_df = filter_meal_candidates_by_calorie_window(d_df, "dinner", dinner_target, 0.80)
	
	b_combos = generate_meal_combinations(b_df, "breakfast", bfast_target)
	l_combos = generate_meal_combinations(l_df, "lunch", lunch_target)
	d_combos = generate_meal_combinations(d_df, "dinner", dinner_target)
	
	all_plans = []
	for r in range(12):
		plans = generate_genetic_algorithm_plans(b_combos, l_combos, d_combos, targets, meal_split, macro_priority, population_size=100, generations=50, top_n=200)
		all_plans.extend(plans)
	
	# Sort by score but add some randomization for variety
	all_plans.sort(key=lambda p: score_meal_plan(p, targets, meal_split, macro_priority), reverse=True)
	
	# Shuffle top plans slightly to ensure variety
	import random
	top_plans = all_plans[:50]  # Top 50 plans
	random.shuffle(top_plans)
	all_plans = top_plans + all_plans[50:]  # Keep top shuffled, rest as is
	
	weekly_plans = []
	used_breakfasts = set()
	used_lunches = set()
	used_dinners = set()
	
	# Track last used meals to prevent consecutive usage
	last_breakfast = None
	last_lunch = None
	last_dinner = None
	
	for day in range(num_days):
		best_plan = None
		best_score = -1
		
		for plan in all_plans:
			if not plan or not plan.get("meals"):
				continue
				
			bfast_meal = plan["meals"].get("breakfast")
			lunch_meal = plan["meals"].get("lunch")
			dinner_meal = plan["meals"].get("dinner")
			
			if not bfast_meal or not lunch_meal or not dinner_meal:
				continue
				
			b_bucket = bfast_meal.get("bucket_id", "")
			l_bucket = lunch_meal.get("bucket_id", "")
			d_bucket = dinner_meal.get("bucket_id", "")
			
			if not b_bucket or not l_bucket or not d_bucket:
				continue
			
			if (b_bucket in used_breakfasts or 
				l_bucket in used_lunches or 
				d_bucket in used_dinners):
				continue
			
			if (last_breakfast and b_bucket == last_breakfast or
				last_lunch and l_bucket == last_lunch or
				last_dinner and d_bucket == last_dinner):
				continue
			
			nutrition_score = score_meal_plan(plan, targets, meal_split, macro_priority)
			
			if nutrition_score > best_score:
				best_score = nutrition_score
				best_plan = plan
		
		if best_plan:
			weekly_plans.append(best_plan)
			bfast_meal = best_plan["meals"].get("breakfast")
			lunch_meal = best_plan["meals"].get("lunch")
			dinner_meal = best_plan["meals"].get("dinner")
			
			if bfast_meal:
				b_bucket = bfast_meal.get("bucket_id", "")
				if b_bucket:
					used_breakfasts.add(b_bucket)
					last_breakfast = b_bucket
					
			if lunch_meal:
				l_bucket = lunch_meal.get("bucket_id", "")
				if l_bucket:
					used_lunches.add(l_bucket)
					last_lunch = l_bucket
					
			if dinner_meal:
				d_bucket = dinner_meal.get("bucket_id", "")
				if d_bucket:
					used_dinners.add(d_bucket)
					last_dinner = d_bucket
		else:
			break
	
	return targets, weekly_plans



@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "GET":
		# Check if loading a saved plan
		load_date = request.args.get('load_date')
		if load_date:
			saved_plan = get_saved_plan(load_date)
			if saved_plan:
				return render_template("mealplan.html",
									plan=saved_plan["plan"],
									targets=get_targets(saved_plan["user_inputs"]),
									user_inputs=saved_plan["user_inputs"],
									plan_index=0,
									total_plans=1,
									breakfast_pct=saved_plan["meal_split"]["breakfast"],
									lunch_pct=saved_plan["meal_split"]["lunch"],
									dinner_pct=saved_plan["meal_split"]["dinner"],
									macro_focus=saved_plan["macro_focus"],
									plan_type="single",
									current_date=load_date,
									error_message=None)
		
		return render_template("mealplan.html",
							plan=None,
							targets=None,
							user_inputs=None,
							plan_index=0,
							total_plans=0,
							breakfast_pct=25,
							lunch_pct=40,
							dinner_pct=35,
							macro_focus="protein",
							plan_type="single",
							current_date=datetime.now().strftime("%Y-%m-%d"),
							error_message=None)

	action = request.form.get("action")
	error_message = None

	try:
		daily_calories_input = request.form.get("daily_calories", "").strip()
		daily_calories = int(daily_calories_input) if daily_calories_input and daily_calories_input.isdigit() else None
		
		user_inputs = {
			"age": int(request.form.get("age")),
			"gender": request.form.get("gender"),
			"height": float(request.form.get("height")),
			"weight": float(request.form.get("weight")),
			"pregnancy": request.form.get("pregnancy") if request.form.get("gender") == "female" else None,
			"lactating": request.form.get("lactating") if request.form.get("gender") == "female" else None,
			"activity": request.form.get("activity"),
			"region": request.form.get("region"),
			"preference": request.form.get("preference"),
			"daily_calories": daily_calories,
	}
		meal_split = {
			"breakfast": int(request.form.get("breakfast_pct")),
			"lunch": int(request.form.get("lunch_pct")),
			"dinner": int(request.form.get("dinner_pct")),
		}
		macro_choice = (request.form.get("macro_focus") or "protein").lower()
		macro_priority = MACRO_FOCUS_MAP.get(macro_choice, MACRO_FOCUS_MAP["protein"])

		# Validate
		if not all([user_inputs["gender"], user_inputs["activity"], user_inputs["region"], user_inputs["preference"]]):
			raise ValueError("Please complete all required fields.")
		if sum(meal_split.values()) != 100 or any(v < 20 for v in meal_split.values()):
			raise ValueError("Meal split must sum to 100% with each at least 20%.")

		plan_type = request.form.get("plan_type", "single")
		
		if plan_type == "weekly":
			targets, plans = generate_weekly_plans(user_inputs, meal_split, macro_priority, num_days=7)
		else:
			targets, plans = generate_plans(user_inputs, meal_split, macro_priority)
			
	except Exception as e:
		error_message = str(e)
		plans = []
		targets = None
		user_inputs = None
		meal_split = {
			"breakfast": int(request.form.get("breakfast_pct") or 25),
			"lunch": int(request.form.get("lunch_pct") or 40),
			"dinner": int(request.form.get("dinner_pct") or 35),
		}
		macro_choice = request.form.get("macro_focus") or "protein"

	plan_index = int(request.form.get("plan_index", 0))

	show_all_plans = False

	if action == "next":
		plan_index = plan_index + 1
	elif action == "previous":
		plan_index = plan_index - 1
	elif action == "show_all":
		show_all_plans = True
	elif action == "restart":
		return redirect(url_for('index'))

	if not plans:
		plan_index = 0
		plan = None
	else:
		plan_index = plan_index % max(1, len(plans))
		plan = plans[plan_index] if plan_index < len(plans) else None

	meal_macros = None
	meal_calorie_percentages = None
	if plan and plan.get("meals") and plan.get("total_macros"):
		bfast_meal = plan["meals"].get("breakfast")
		lunch_meal = plan["meals"].get("lunch")
		dinner_meal = plan["meals"].get("dinner")
		
		if bfast_meal and lunch_meal and dinner_meal:
			meal_macros = {
				"breakfast": bfast_meal.get("macros", {}),
				"lunch": lunch_meal.get("macros", {}),
				"dinner": dinner_meal.get("macros", {}),
				"total": plan["total_macros"],
			}
			
			cal_b = meal_macros["breakfast"].get("Calories", 0)
			cal_l = meal_macros["lunch"].get("Calories", 0)
			cal_d = meal_macros["dinner"].get("Calories", 0)
			total_cals = cal_b + cal_l + cal_d
			
			if total_cals > 0:
				meal_calorie_percentages = {
					"breakfast": {"actual": round((cal_b/total_cals)*100, 1), "target": meal_split["breakfast"]},
					"lunch": {"actual": round((cal_l/total_cals)*100, 1), "target": meal_split["lunch"]},
					"dinner": {"actual": round((cal_d/total_cals)*100, 1), "target": meal_split["dinner"]},
				}

	return render_template("mealplan.html",
							plan=plan,
							targets=targets,
							meal_macros=meal_macros,
							user_inputs=user_inputs,
							plan_index=plan_index,
							total_plans=len(plans),
							breakfast_pct=meal_split["breakfast"],
							lunch_pct=meal_split["lunch"],
							dinner_pct=meal_split["dinner"],
							macro_focus=macro_choice,
							meal_calorie_percentages=meal_calorie_percentages,
							plan_type=plan_type,
							show_all_plans=show_all_plans,
							all_plans=plans if show_all_plans else None,
							current_date=datetime.now().strftime("%Y-%m-%d"),
							error_message=error_message)


@app.route("/compare", methods=["GET", "POST"])
def compare_algorithms_route():
	if request.method == "GET":
		return render_template("mealplan.html",
							plan=None,
							targets=None,
							user_inputs=None,
							plan_index=0,
							total_plans=0,
							breakfast_pct=25,
							lunch_pct=40,
							dinner_pct=35,
							macro_focus="protein",
							plan_type="single",
							current_date=datetime.now().strftime("%Y-%m-%d"),
							error_message=None)

	try:
		daily_calories_input = request.form.get("daily_calories", "").strip()
		daily_calories = int(daily_calories_input) if daily_calories_input and daily_calories_input.isdigit() else None
		
		user_inputs = {
			"age": int(request.form.get("age")),
			"gender": request.form.get("gender"),
			"height": float(request.form.get("height")),
			"weight": float(request.form.get("weight")),
			"pregnancy": request.form.get("pregnancy") if request.form.get("gender") == "female" else None,
			"lactating": request.form.get("lactating") if request.form.get("gender") == "female" else None,
			"activity": request.form.get("activity"),
			"region": request.form.get("region"),
			"preference": request.form.get("preference"),
			"daily_calories": daily_calories,
		}
		meal_split = {
			"breakfast": int(request.form.get("breakfast_pct")),
			"lunch": int(request.form.get("lunch_pct")),
			"dinner": int(request.form.get("dinner_pct")),
		}
		macro_choice = (request.form.get("macro_focus") or "protein").lower()
		macro_priority = MACRO_FOCUS_MAP.get(macro_choice, MACRO_FOCUS_MAP["protein"])

		# Validate
		if not all([user_inputs["gender"], user_inputs["activity"], user_inputs["region"], user_inputs["preference"]]):
			raise ValueError("Please complete all required fields.")
		if sum(meal_split.values()) != 100 or any(v < 20 for v in meal_split.values()):
			raise ValueError("Meal split must sum to 100% with each at least 20%.")

		targets = get_targets(user_inputs)
		bfast_target = (targets["TDEE"] * meal_split["breakfast"]) / 100
		lunch_target = (targets["TDEE"] * meal_split["lunch"]) / 100
		dinner_target = (targets["TDEE"] * meal_split["dinner"]) / 100

		b_df = load_dataset("Breakfast_Adjusted_Max.xlsx", user_inputs.get("region"), user_inputs.get("preference"))
		l_df = load_dataset("Lunch_Adjusted_Max.xlsx", user_inputs.get("region"), user_inputs.get("preference"))
		d_df = load_dataset("Dinner_Adjusted_Max.xlsx", user_inputs.get("region"), user_inputs.get("preference"))
		
		b_df = filter_meal_candidates_by_calorie_window(b_df, "breakfast", bfast_target, 0.80)
		l_df = filter_meal_candidates_by_calorie_window(l_df, "lunch", lunch_target, 0.80)
		d_df = filter_meal_candidates_by_calorie_window(d_df, "dinner", dinner_target, 0.80)
		
		b_combos = generate_meal_combinations(b_df, "breakfast", bfast_target)
		l_combos = generate_meal_combinations(l_df, "lunch", lunch_target)
		d_combos = generate_meal_combinations(d_df, "dinner", dinner_target)
		
		# Run algorithm comparison
		comparison_metrics = compare_algorithms(b_combos, l_combos, d_combos, targets, meal_split, macro_priority)
		
		# Get best plan from each algorithm
		best_brute = comparison_metrics['brute_force'].get('best_plan', None)
		best_genetic = comparison_metrics['genetic'].get('best_plan', None)
		
		return render_template("algorithm_comparison.html",
							targets=targets,
							user_inputs=user_inputs,
							comparison_metrics=comparison_metrics,
							best_brute=best_brute,
							best_genetic=best_genetic,
							meal_split=meal_split,
							macro_focus=macro_choice)
							
	except Exception as e:
		error_message = str(e)
		return render_template("mealplan.html",
							plan=None,
							targets=None,
							user_inputs=None,
							plan_index=0,
							total_plans=0,
							breakfast_pct=25,
							lunch_pct=40,
							dinner_pct=35,
							macro_focus="protein",
							plan_type="single",
							current_date=datetime.now().strftime("%Y-%m-%d"),
							error_message=error_message)

@app.route("/save_plan", methods=["POST"])
def save_plan():
	"""Save a meal plan for a specific date"""
	try:
		data = request.get_json()
		plan_date = data.get('date')
		plan_data = data.get('plan')
		user_inputs = data.get('user_inputs')
		meal_split = data.get('meal_split')
		macro_focus = data.get('macro_focus')
		
		if not all([plan_date, plan_data, user_inputs, meal_split, macro_focus]):
			return jsonify({"success": False, "message": "Missing required data"})
		
		success = save_meal_plan(plan_date, plan_data, user_inputs, meal_split, macro_focus)
		
		if success:
			return jsonify({"success": True, "message": f"Meal plan saved for {plan_date}"})
		else:
			return jsonify({"success": False, "message": "Failed to save meal plan"})
			
	except Exception as e:
		return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route("/load_plan/<plan_date>")
def load_plan(plan_date):
	"""Load a saved meal plan for a specific date"""
	try:
		saved_plan = get_saved_plan(plan_date)
		
		if saved_plan:
			return jsonify({
				"success": True,
				"plan": saved_plan["plan"],
				"user_inputs": saved_plan["user_inputs"],
				"meal_split": saved_plan["meal_split"],
				"macro_focus": saved_plan["macro_focus"],
				"saved_at": saved_plan["saved_at"]
			})
		else:
			return jsonify({"success": False, "message": "No plan found for this date"})
			
	except Exception as e:
		return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route("/saved_dates")
def get_saved_dates():
	"""Get all dates with saved meal plans"""
	try:
		dates = get_all_saved_dates()
		return jsonify({"success": True, "dates": dates})
	except Exception as e:
		return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route("/saved_plans")
def saved_plans_page():
	"""Page to view all saved meal plans"""
	saved_dates = get_all_saved_dates()
	return render_template("saved_plans.html", saved_dates=saved_dates)

@app.route("/genetic_visualization")
def genetic_visualization():
	"""Page to visualize genetic algorithm step by step"""
	# Create a simple example for visualization
	example_data = {
		"user_inputs": {
			"age": 25,
			"gender": "Male",
			"weight": 70,
			"height": 175,
			"activity": "Moderately Active",
			"goal": "Maintain Weight",
			"region": "Indian",
			"preference": "Non-Vegetarian"
		},
		"meal_split": {
			"breakfast": 25,
			"lunch": 40,
			"dinner": 35
		},
		"macro_focus": "Balanced"
	}
	
	# Generate targets
	targets = get_targets(example_data["user_inputs"])
	
	# Create example meal combinations (top 10 each)
	bfast_combos = [
		{"name": "Oatmeal with Fruits", "calories": 300, "protein": 12, "carbs": 45, "fat": 8},
		{"name": "Scrambled Eggs", "calories": 280, "protein": 18, "carbs": 2, "fat": 22},
		{"name": "Pancakes", "calories": 320, "protein": 8, "carbs": 52, "fat": 10},
		{"name": "Toast with Butter", "calories": 250, "protein": 6, "carbs": 35, "fat": 12},
		{"name": "Cereal with Milk", "calories": 290, "protein": 10, "carbs": 48, "fat": 6},
		{"name": "Yogurt Parfait", "calories": 270, "protein": 15, "carbs": 35, "fat": 8},
		{"name": "Smoothie Bowl", "calories": 310, "protein": 12, "carbs": 42, "fat": 9},
		{"name": "Bagel with Cream Cheese", "calories": 340, "protein": 11, "carbs": 45, "fat": 14},
		{"name": "French Toast", "calories": 330, "protein": 9, "carbs": 38, "fat": 16},
		{"name": "Breakfast Burrito", "calories": 380, "protein": 16, "carbs": 35, "fat": 18}
	]
	
	lunch_combos = [
		{"name": "Chicken Salad", "calories": 450, "protein": 35, "carbs": 20, "fat": 25},
		{"name": "Rice with Curry", "calories": 480, "protein": 18, "carbs": 65, "fat": 15},
		{"name": "Pasta with Meatballs", "calories": 520, "protein": 28, "carbs": 55, "fat": 18},
		{"name": "Grilled Chicken Sandwich", "calories": 420, "protein": 32, "carbs": 35, "fat": 16},
		{"name": "Fish and Rice", "calories": 460, "protein": 30, "carbs": 50, "fat": 12},
		{"name": "Vegetable Stir Fry", "calories": 380, "protein": 15, "carbs": 45, "fat": 18},
		{"name": "Burger with Fries", "calories": 650, "protein": 25, "carbs": 70, "fat": 28},
		{"name": "Quinoa Bowl", "calories": 440, "protein": 20, "carbs": 55, "fat": 14},
		{"name": "Taco Salad", "calories": 480, "protein": 22, "carbs": 40, "fat": 25},
		{"name": "Soup and Bread", "calories": 400, "protein": 18, "carbs": 45, "fat": 16}
	]
	
	dinner_combos = [
		{"name": "Grilled Salmon", "calories": 420, "protein": 35, "carbs": 8, "fat": 25},
		{"name": "Pasta Carbonara", "calories": 580, "protein": 22, "carbs": 65, "fat": 24},
		{"name": "Chicken Curry", "calories": 520, "protein": 32, "carbs": 45, "fat": 22},
		{"name": "Beef Steak", "calories": 450, "protein": 40, "carbs": 5, "fat": 28},
		{"name": "Vegetable Lasagna", "calories": 480, "protein": 20, "carbs": 55, "fat": 18},
		{"name": "Fish Tacos", "calories": 440, "protein": 28, "carbs": 35, "fat": 20},
		{"name": "Pizza Slice", "calories": 350, "protein": 15, "carbs": 40, "fat": 16},
		{"name": "Stir Fry Noodles", "calories": 420, "protein": 18, "carbs": 50, "fat": 16},
		{"name": "Roasted Chicken", "calories": 380, "protein": 35, "carbs": 12, "fat": 22},
		{"name": "Vegetable Curry", "calories": 400, "protein": 15, "carbs": 45, "fat": 18}
	]
	
	return render_template("genetic_visualization.html", 
	                     targets=targets, 
	                     bfast_combos=bfast_combos,
	                     lunch_combos=lunch_combos, 
	                     dinner_combos=dinner_combos,
	                     meal_split=example_data["meal_split"],
	                     macro_focus=example_data["macro_focus"])

if __name__ == "__main__":
	# Get port from environment variable or default to 5000
	port = int(os.environ.get('PORT', 5000))
	# Set debug to False for production
	debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
	app.run(host='0.0.0.0', port=port, debug=debug)