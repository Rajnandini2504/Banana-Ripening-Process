from . import config

CLASS_NAMES = ["green", "natural", "chemical"]
STAGES = ["GREEN / UNRIPE", "NATURALLY RIPENED", "CHEMICALLY RIPENED"]
RECOMMENDATIONS = {
	"GREEN / UNRIPE": "Allow the banana to ripen naturally before consumption.",
	"NATURALLY RIPENED": "Model predicts natural ripening. This is an image-based estimate.",
	"CHEMICALLY RIPENED": "Model predicts chemical ripening. Further laboratory testing is recommended for confirmation.",
}
DESCRIPTIONS = {
	"GREEN / UNRIPE": "Banana appears to be unripe based on the analyzed image features.",
	"NATURALLY RIPENED": "Banana is predicted as naturally ripened based on the learned image features.",
	"CHEMICALLY RIPENED": "Banana is predicted as chemically ripened based on the learned image features.",
}


def classify_ripening_method(features):
	"""Combine spot, color-distribution, and texture evidence."""
	spot_percentage = float(features.get("brown_spot_percentage", 0.0))
	spot_count = int(features.get("spot_count", 0))
	color_variation = float(features.get("green", 0.0)) + float(features.get("brown", 0.0)) + float(features.get("dark", 0.0))
	texture_entropy = float(features.get("texture_entropy", 0.0))
	contrast = float(features.get("glcm_contrast", 0.0))
	homogeneity = float(features.get("glcm_homogeneity", 0.0))
	energy = float(features.get("glcm_energy", 0.0))
	texture_score = float(features.get("texture_score", 0.0))
	yellow = float(features.get("yellow", 0.0))

	natural_evidence = sum((
		spot_percentage >= config.NATURAL_SPOT_THRESHOLD,
		spot_count >= config.NATURAL_SPOT_COUNT_THRESHOLD,
		color_variation >= config.NATURAL_COLOR_VARIATION_THRESHOLD,
		texture_entropy >= config.NATURAL_TEXTURE_ENTROPY_THRESHOLD,
		contrast >= config.NATURAL_GLCM_CONTRAST_THRESHOLD,
		texture_score >= config.NATURAL_TEXTURE_SCORE_THRESHOLD,
	))
	chemical_evidence = sum((
		spot_percentage <= config.CHEMICAL_SPOT_THRESHOLD,
		spot_count <= config.CHEMICAL_SPOT_COUNT_THRESHOLD,
		yellow >= config.HIGH_YELLOW_PERCENTAGE,
		color_variation <= config.CHEMICAL_COLOR_VARIATION_THRESHOLD,
		texture_entropy <= config.CHEMICAL_TEXTURE_ENTROPY_THRESHOLD,
		contrast <= config.CHEMICAL_GLCM_CONTRAST_THRESHOLD,
		homogeneity >= config.CHEMICAL_GLCM_HOMOGENEITY_THRESHOLD,
		energy >= config.CHEMICAL_GLCM_ENERGY_THRESHOLD,
	))

	if natural_evidence >= config.NATURAL_MIN_EVIDENCE and natural_evidence >= chemical_evidence:
		return "natural"
	if chemical_evidence >= config.CHEMICAL_MIN_EVIDENCE:
		return "chemical"
	return "natural"


def classify_rule_based(features):
	"""Provide an image-based fallback until a real three-class model is trained."""
	green, yellow = features["green"], features["yellow"]
	brown_area = features["brown_area"]
	if green >= 58 and yellow < 25:
		label = 0
	elif brown_area >= 20 and yellow >= 12:
		label = 2
	else:
		label = 1
	stage = STAGES[label]
	return {
		"stage": stage,
		"stage_number": label + 1,
		"confidence": None,
		"recommendation": RECOMMENDATIONS[stage],
		"description": DESCRIPTIONS[stage].replace("learned", "analyzed"),
		"mode": "Image-based classification",
		"probabilities": None,
	}
