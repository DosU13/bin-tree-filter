"""Configuration constants for the Kyrgyz ornamental pattern generator."""

NUM_COLORS = 3
MAX_DEPTH = 10
MIN_DEPTH = 0
HOMOGENEITY_THRESHOLD = 0.98
ORNAMENT_FILENAME = "ornam.jpg"
BACKGROUND_COLOR = (181, 217, 253)

# Image calibration (applied before posterization)
BRIGHTNESS = 0      # additive, -100 to +100
CONTRAST = 1.0      # multiplicative, 0.5 to 3.0
SATURATION = 1.0    # multiplicative, 0.0 to 3.0
