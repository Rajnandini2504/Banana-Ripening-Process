from src.classifier import classify_ripening_method


def _features(**overrides):
	features = {
		"green": 0,
		"yellow": 65,
		"brown": 0,
		"dark": 0,
		"brown_spot_percentage": 12,
		"spot_count": 25,
		"texture_entropy": 6,
		"glcm_contrast": 2,
		"glcm_homogeneity": 0.3,
		"glcm_energy": 0.2,
		"texture_score": 1.1,
	}
	features.update(overrides)
	return features


def test_natural_profiles_use_spots_and_texture_evidence():
	profiles = (
		_features(yellow=65, brown_spot_percentage=12, spot_count=25),
		_features(yellow=70, brown_spot_percentage=8, spot_count=15),
		_features(yellow=60, brown_spot_percentage=10, spot_count=20),
	)

	assert all(classify_ripening_method(profile) == "natural" for profile in profiles)


def test_clean_uniform_yellow_profiles_classify_as_chemical():
	profiles = (
		_features(yellow=95, brown_spot_percentage=0.5, spot_count=2, texture_entropy=3.5, glcm_contrast=0.1, glcm_homogeneity=0.9, glcm_energy=0.8, texture_score=0.2),
		_features(yellow=92, brown_spot_percentage=1, spot_count=3, texture_entropy=3.5, glcm_contrast=0.1, glcm_homogeneity=0.9, glcm_energy=0.8, texture_score=0.2),
		_features(yellow=98, brown_spot_percentage=0, spot_count=0, texture_entropy=3.5, glcm_contrast=0.1, glcm_homogeneity=0.9, glcm_energy=0.8, texture_score=0.2),
	)

	assert all(classify_ripening_method(profile) == "chemical" for profile in profiles)


def test_yellow_percentage_alone_does_not_trigger_chemical_result():
	profile = _features(
		yellow=95,
		brown_spot_percentage=4,
		spot_count=5,
		texture_entropy=5,
		glcm_contrast=0.7,
		glcm_homogeneity=0.5,
		glcm_energy=0.4,
		texture_score=0.6,
	)

	assert classify_ripening_method(profile) == "natural"