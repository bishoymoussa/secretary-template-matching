import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats, signal
from sklearn.metrics import precision_recall_curve, auc, f1_score
import argparse
from tqdm import tqdm
import requests
import io
import os
from sklearn.preprocessing import StandardScaler

class OnlineTemplateMatching:
    """
    Implementation of Online Template Matching using Optimal Stopping Theory.
    
    This algorithm adapts the secretary problem approach to template matching:
    1. Observe a portion of the data stream without making decisions
    2. Use this observation to establish dynamic thresholds
    3. Make immediate (online) match decisions for the remainder of the stream
    """
    
    def __init__(self, template, observation_ratio=0.368, 
                 adaptation_rate=0.1, confidence_level=1.5):
        """
        Initialize the online template matching algorithm.
        
        Parameters:
        -----------
        template : array-like
            The template pattern to match against
        observation_ratio : float
            Ratio of the stream to observe before making decisions (1/e ≈ 0.368 is optimal from secretary problem)
        adaptation_rate : float
            Rate at which threshold adapts over time (β in the theory)
        confidence_level : float
            Number of standard deviations for confidence bounds (k in the theory)
        """
        self.template = np.array(template)
        self.template_len = len(template)
        self.observation_ratio = observation_ratio
        self.adaptation_rate = adaptation_rate
        self.confidence_level = confidence_level
        
        # Statistics tracking
        self.similarity_scores = []
        self.threshold_values = []
        self.threshold = None
        self.initial_threshold = None
        self.mean_score = 0
        self.std_score = 1
        self.matches = []
        self.position = 0
        self.observation_phase = True
        self.observation_end = 0
        
    def similarity(self, window):
        """
        Calculate similarity between window and template.
        Uses normalized cross-correlation as a similarity measure.
        
        Parameters:
        -----------
        window : array-like
            Window of data to compare against template
            
        Returns:
        --------
        float : Similarity score (higher means more similar, range [0,1])
        """
        # Handle edge cases
        if len(window) != self.template_len:
            raise ValueError("Window length must match template length")
            
        # Use scipy's signal correlation for better numerical stability
        correlation = signal.correlate(
            window - np.mean(window), 
            self.template - np.mean(self.template), 
            mode='valid'
        )
        
        # Normalize by standard deviations
        window_std = np.std(window)
        template_std = np.std(self.template)
        
        if window_std < 1e-10 or template_std < 1e-10:
            return 0.0
            
        # Normalize and scale to [0, 1]
        normalized_corr = correlation[0] / (window_std * template_std * self.template_len)
        similarity = (normalized_corr + 1) / 2
        
        return similarity
    
    def process_window(self, window):
        """
        Process a new window of data and decide if it matches the template.
        
        Parameters:
        -----------
        window : array-like
            New window of data
            
        Returns:
        --------
        bool : True if match is detected, False otherwise
        """
        self.position += 1
        similarity_score = self.similarity(window)
        self.similarity_scores.append(similarity_score)
        
        # Still in observation phase - don't make match decisions yet
        if self.observation_phase:
            if self.position >= self.observation_end:
                self.finish_observation_phase()
            self.threshold_values.append(None)
            return False
        
        # In selection phase, check for match using dynamic threshold
        match_detected = self.is_match(similarity_score, self.position)
        return match_detected
    
    def start_observation(self, expected_length):
        """
        Begin the observation phase.
        
        Parameters:
        -----------
        expected_length : int
            Expected length of the data stream
        """
        self.observation_end = max(int(expected_length * self.observation_ratio), 1)
        self.observation_phase = True
        self.position = 0
        self.similarity_scores = []
        self.threshold_values = []
        self.matches = []
    
    def finish_observation_phase(self):
        """Finalize the observation phase and prepare for selection phase."""
        self.observation_phase = False
        
        if len(self.similarity_scores) > 0:
            self.mean_score = np.mean(self.similarity_scores)
            self.std_score = max(np.std(self.similarity_scores), 1e-10)
            
            # Use 95th percentile for initial threshold - less strict
            if len(self.similarity_scores) > 10:
                self.initial_threshold = np.percentile(self.similarity_scores, 95)
            else:
                # Use mean + 2*std as fallback
                self.initial_threshold = self.mean_score + 2 * self.std_score
            
            self.threshold = self.initial_threshold
    
    def is_match(self, score, position):
        """
        Determine if the current score represents a match using dynamic thresholding.
        """
        if self.observation_phase:
            return False
        
        # Start with the initial threshold
        dynamic_threshold = self.initial_threshold
        
        # Statistical confidence bound based on distribution of scores
        statistical_threshold = self.mean_score + self.confidence_level * self.std_score
        
        # Apply dynamic threshold adaptation with slower decay
        if position > self.observation_end:
            relative_position = position / self.observation_end
            # Reduce adaptation rate over time to prevent excessive threshold decrease
            current_adaptation_rate = self.adaptation_rate / (1 + np.log10(relative_position))
            adaptation = current_adaptation_rate * np.log(relative_position)
            dynamic_threshold -= adaptation
        
        # Use weighted combination of both thresholds
        # Give more weight to statistical threshold for stability
        weight = 0.7  # 70% statistical, 30% dynamic
        effective_threshold = weight * statistical_threshold + (1 - weight) * dynamic_threshold
        
        # Add minimum threshold floor to prevent excessive false positives
        min_threshold = self.mean_score + 0.5 * self.std_score
        effective_threshold = max(effective_threshold, min_threshold)
        
        # Record the threshold for this position
        self.threshold = effective_threshold
        self.threshold_values.append(effective_threshold)
        
        return score > effective_threshold
    
    def process_stream(self, stream):
        """
        Process an entire data stream.
        
        Parameters:
        -----------
        stream : array-like
            Data stream to process
            
        Returns:
        --------
        list : Positions of detected matches
        """
        stream_len = len(stream)
        self.start_observation(stream_len - self.template_len + 1)
        
        matches = []
        for i in tqdm(range(stream_len - self.template_len + 1), desc="Processing stream"):
            window = stream[i:i+self.template_len]
            is_match = self.process_window(window)
            if is_match:
                matches.append(i)
        
        return matches


def traditional_template_matching(stream, template, threshold=0.65):
    """
    Implement a traditional template matching approach for comparison.
    Uses a fixed threshold approach with the same similarity measure.
    
    Parameters:
    -----------
    stream : array-like
        Data stream to process
    template : array-like
        Template pattern to match
    threshold : float
        Fixed similarity threshold for matches
        
    Returns:
    --------
    list : Positions of detected matches
    """
    template_len = len(template)
    matches = []
    similarity_scores = []
    
    # Normalize template for faster computation
    template_norm = template - np.mean(template)
    template_std = np.std(template)
    
    # Sliding window approach with fixed threshold
    for i in tqdm(range(len(stream) - template_len + 1), desc="Traditional matching"):
        window = stream[i:i+template_len]
        
        # Normalize window
        window_norm = window - np.mean(window)
        window_std = np.std(window)
        
        if window_std < 1e-10 or template_std < 1e-10:
            similarity = 0.0
        else:
            # Compute correlation
            correlation = np.sum(window_norm * template_norm) / (window_std * template_std * template_len)
            similarity = (correlation + 1) / 2
        
        similarity_scores.append(similarity)
        
        if similarity > threshold:
            matches.append(i)
    
    return matches, similarity_scores


def download_ucr_dataset(dataset_name="ECG5000"):
    """
    Download a dataset from the UCR Time Series Archive.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to download
        
    Returns:
    --------
    tuple : (train_data, test_data, train_labels, test_labels)
    """
    print(f"Downloading UCR dataset: {dataset_name}")
    
    # For Ford A dataset, we'll use a specific GitHub repo
    if dataset_name == "FordA":
        base_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        
        try:
            # Get the train data
            train_url = f"{base_url}FordA_TRAIN.tsv"
            response = requests.get(train_url, timeout=30)
            response.raise_for_status()
            
            train_data = pd.read_csv(io.StringIO(response.text), sep='\t', header=None)
            
            # Get the test data
            test_url = f"{base_url}FordA_TEST.tsv"
            response = requests.get(test_url, timeout=30)
            response.raise_for_status()
            
            test_data = pd.read_csv(io.StringIO(response.text), sep='\t', header=None)
            
            # Extract labels and data
            train_labels = train_data.iloc[:, 0].values
            train_data = train_data.iloc[:, 1:].values
            
            test_labels = test_data.iloc[:, 0].values
            test_data = test_data.iloc[:, 1:].values
            
            print(f"Successfully downloaded: train_data shape: {train_data.shape}, test_data shape: {test_data.shape}")
            return train_data, test_data, train_labels, test_labels
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Using synthetic data instead")
            # Generate synthetic data as fallback
            return generate_synthetic_data()
    else:
        # For other datasets, we could implement similar downloading logic
        print(f"Dataset {dataset_name} not directly supported. Using synthetic data instead.")
        return generate_synthetic_data()


def generate_synthetic_data():
    """
    Generate synthetic time series data as a fallback.
    
    Returns:
    --------
    tuple : (train_data, test_data, train_labels, test_labels)
    """
    # Create a synthetic pattern
    x = np.linspace(0, 2*np.pi, 500)
    pattern = np.sin(x) + 0.5 * np.sin(3*x)
    
    # Create training data: 100 normal and 100 abnormal patterns
    np.random.seed(42)
    normal_noise = 0.2
    abnormal_noise = 0.1
    
    # Normal patterns (random noise)
    normal_patterns = np.random.randn(100, 500) * normal_noise
    
    # Abnormal patterns (pattern with less noise)
    abnormal_patterns = np.tile(pattern, (100, 1)) + np.random.randn(100, 500) * abnormal_noise
    
    # Combine data and create labels
    train_data = np.vstack((normal_patterns, abnormal_patterns))
    train_labels = np.hstack((np.ones(100) * -1, np.ones(100)))
    
    # Create test data (similar to training)
    normal_test = np.random.randn(50, 500) * normal_noise
    abnormal_test = np.tile(pattern, (50, 1)) + np.random.randn(50, 500) * abnormal_noise
    
    test_data = np.vstack((normal_test, abnormal_test))
    test_labels = np.hstack((np.ones(50) * -1, np.ones(50)))
    
    return train_data, test_data, train_labels, test_labels


def prepare_ucr_data_for_template_matching(dataset_name="FordA"):
    """
    Prepare UCR data for template matching with improved processing.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to use
        
    Returns:
    --------
    tuple : (stream, template, true_matches)
    """
    # Download the data
    train_data, test_data, train_labels, test_labels = download_ucr_dataset(dataset_name)
    
    # For each class, pick a representative as a template
    unique_labels = np.unique(train_labels)
    print(f"Dataset has {len(unique_labels)} classes: {unique_labels}")
    
    # Choose a class to use for testing template matching (often class 1 is the abnormal/positive class)
    class_to_use = 1
    
    # Get all examples of this class from the training set
    class_indices = np.where(train_labels == class_to_use)[0]
    
    if len(class_indices) == 0:
        print(f"No examples of class {class_to_use} found. Using class {unique_labels[0]} instead.")
        class_to_use = unique_labels[0]
        class_indices = np.where(train_labels == class_to_use)[0]
    
    print(f"Using class {class_to_use} with {len(class_indices)} examples")
    
    # IMPROVEMENT 1: Use average of multiple examples as template for robustness
    num_templates_to_avg = min(10, len(class_indices))
    template_indices = np.random.choice(class_indices, num_templates_to_avg, replace=False)
    template = np.mean(train_data[template_indices], axis=0)
    
    # IMPROVEMENT 2: Apply smoothing to reduce noise
    template = signal.savgol_filter(template, 11, 3)
    
    # IMPROVEMENT 3: Create a longer test stream with multiple known template positions
    # Create stream from test data with injected templates
    stream_length = min(10000, test_data.shape[0] * test_data.shape[1])
    
    # Use test data to create a baseline stream
    other_class_indices = np.where(test_labels != class_to_use)[0]
    if len(other_class_indices) < 5:
        other_class_indices = np.arange(len(test_labels))
    
    # Concatenate enough test samples to reach desired length
    stream_parts = []
    while sum(len(p) for p in stream_parts) < stream_length:
        idx = np.random.choice(other_class_indices)
        stream_parts.append(test_data[idx])
    
    # Combine parts and trim to desired length
    stream = np.concatenate(stream_parts)[:stream_length]
    
    # IMPROVEMENT 4: Insert templates with controlled SNR
    template_len = len(template)
    num_templates = 10
    positions = np.linspace(0, len(stream) - template_len - 1, num_templates + 2)[1:-1]
    positions = positions.astype(int)
    
    true_matches = []
    for i, pos in enumerate(positions):
        # Scale to match local variance
        local_std = np.std(stream[pos:pos+template_len])
        template_std = np.std(template)
        scaling_factor = max(local_std / template_std * 1.5, 1.0)
        
        # Insert with additive combination (weighted average)
        weight = 0.7  # Control visibility of template
        stream[pos:pos+template_len] = (
            weight * (template * scaling_factor) + 
            (1 - weight) * stream[pos:pos+template_len]
        )
        true_matches.append(pos)
    
    print(f"Created stream of length {len(stream)} with {len(true_matches)} true matches")
    print(f"True match positions: {true_matches}")
    
    # Plot an example of a true match
    plt.figure(figsize=(10, 6))
    plt.plot(stream[true_matches[0]:true_matches[0]+template_len], label='Stream with Template')
    plt.plot(template, label='Template', alpha=0.7)
    plt.title('Example of Template in Stream')
    plt.legend()
    plt.savefig('template_example.png')
    
    return stream, template, true_matches


def evaluate_performance(detected_matches, true_matches, stream_length, window_size=None):
    """
    Evaluate the performance of template matching with improved tolerance.
    
    Parameters:
    -----------
    detected_matches : list
        Positions of detected matches
    true_matches : list
        Positions of true matches
    stream_length : int
        Length of the data stream
    window_size : int, optional
        Size of window to consider a match correct
        
    Returns:
    --------
    dict : Performance metrics
    """
    if window_size is None:
        # IMPROVEMENT: Use 20% of template length as tolerance
        window_size = 100  # Larger tolerance window for UCR datasets
    
    # Create binary arrays for precision-recall analysis
    true_positives_arr = np.zeros(stream_length, dtype=bool)
    predicted_positives_arr = np.zeros(stream_length, dtype=bool)
    
    # Mark true matches with window tolerance
    for pos in true_matches:
        start = max(0, pos - window_size // 2)
        end = min(stream_length, pos + window_size // 2 + 1)
        true_positives_arr[start:end] = True
    
    # Mark predicted matches with window tolerance
    for pos in detected_matches:
        if pos < stream_length:  # Ensure valid position
            start = max(0, pos - window_size // 2)
            end = min(stream_length, pos + window_size // 2 + 1)
            predicted_positives_arr[start:end] = True
    
    # Calculate metrics
    true_positive_count = np.sum(predicted_positives_arr & true_positives_arr)
    false_positive_count = np.sum(predicted_positives_arr & ~true_positives_arr)
    false_negative_count = np.sum(~predicted_positives_arr & true_positives_arr)
    
    # Handle division by zero
    precision = true_positive_count / max(true_positive_count + false_positive_count, 1)
    recall = true_positive_count / max(true_positive_count + false_negative_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    # Statistical significance test (binomial test)
    random_match_prob = np.sum(true_positives_arr) / stream_length
    total_trials = true_positive_count + false_positive_count
    
    # Only perform binomial test if we have valid data
    if total_trials > 0:
        p_value = stats.binomtest(true_positive_count, 
                               total_trials,
                               p=random_match_prob, 
                               alternative='greater').pvalue
    else:
        p_value = 1.0  # No statistical significance if no matches found
    
    # Record actual match indices for detailed analysis
    true_indices = []
    for pos in detected_matches:
        for true_pos in true_matches:
            if abs(pos - true_pos) <= window_size // 2:
                true_indices.append(pos)
                break
    
    results = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "True Positives": true_positive_count,
        "False Positives": false_positive_count,
        "False Negatives": false_negative_count,
        "True Indices": true_indices,
        "p-value": p_value,
        "Window Size": window_size
    }
    
    return results


def visualize_results(stream, template, online_matches, trad_matches, true_matches,
                     online_scores, trad_scores, online_thresholds, title="Template Matching Results"):
    """
    Visualize the results of template matching for both methods with improved visualization.
    
    Parameters:
    -----------
    stream : array-like
        Data stream
    template : array-like
        Template pattern
    online_matches : list
        Positions of matches detected by online method
    trad_matches : list
        Positions of matches detected by traditional method
    true_matches : list
        Positions of true matches
    online_scores : list
        Similarity scores from online method
    trad_scores : list
        Similarity scores from traditional method
    online_thresholds : list
        Threshold values from online method
    title : str
        Plot title
    """
    template_len = len(template)
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # Calculate error metrics for detected matches
    def calc_match_error(matches):
        if not matches:
            return "N/A"
        errors = []
        for m in matches:
            distances = [abs(m - t) for t in true_matches]
            min_dist = min(distances) if distances else float('inf')
            if min_dist < template_len:  # Only count as a hit if within range
                errors.append(min_dist)
        return np.mean(errors) if errors else "N/A"
    
    online_error = calc_match_error(online_matches)
    trad_error = calc_match_error(trad_matches)
    
    # 1. Data stream overview (zoomed out)
    axs[0].plot(stream, 'b-', alpha=0.7, label='Data Stream')
    axs[0].set_title(f'Full Data Stream (Online Avg Error: {online_error}, Trad Avg Error: {trad_error})')
    
    # Mark match regions
    for pos in true_matches:
        if pos < len(stream) - template_len:
            axs[0].axvspan(pos, pos + template_len, color='g', alpha=0.2)
    
    # Mark detected matches
    for pos in online_matches:
        if pos < len(stream) - template_len:
            axs[0].axvline(x=pos, color='r', linestyle='--', alpha=0.4)
    
    for pos in trad_matches:
        if pos < len(stream) - template_len:
            axs[0].axvline(x=pos, color='m', linestyle=':', alpha=0.4)
    
    axs[0].legend(['Data Stream', 'True Match Regions', 'Online Detections', 'Traditional Detections'])
    
    # 2. Similarity scores and thresholds
    x_vals = range(len(online_scores))
    axs[1].plot(x_vals, online_scores, 'r-', alpha=0.7, label='Online Method Scores')
    axs[1].plot(x_vals, trad_scores, 'm-', alpha=0.7, label='Traditional Method Scores')
    
    # Plot online dynamic threshold
    valid_thresholds = [(i, t) for i, t in enumerate(online_thresholds) if t is not None]
    if valid_thresholds:
        threshold_x, threshold_y = zip(*valid_thresholds)
        axs[1].plot(threshold_x, threshold_y, 'g-', label='Online Dynamic Threshold')
    
    # Add horizontal line for traditional threshold
    trad_threshold = 0.65  # Adjusted threshold
    axs[1].axhline(y=trad_threshold, color='b', linestyle='--', 
                  label='Traditional Fixed Threshold')
    
    # Mark true match positions on similarity plot
    for pos in true_matches:
        if pos < len(online_scores):
            axs[1].axvline(x=pos, color='g', linestyle='-', alpha=0.3)
    
    axs[1].set_title('Similarity Scores and Thresholds')
    axs[1].set_ylabel('Similarity Score')
    axs[1].legend()
    
    # 3. Zoomed view of a template match (if available)
    if true_matches and true_matches[0] < len(stream) - template_len:
        # Choose the first true match for detailed view
        zoom_pos = true_matches[0]
        zoom_window = 2 * template_len
        
        start = max(0, zoom_pos - zoom_window//4)
        end = min(len(stream), zoom_pos + template_len + zoom_window//4)
        
        axs[2].plot(range(start, end), stream[start:end], 'b-', label='Stream')
        
        # Overlay the template at the match position
        template_x = range(zoom_pos, zoom_pos + template_len)
        template_scaled = template * np.std(stream[zoom_pos:zoom_pos+template_len]) / np.std(template)
        template_shifted = template_scaled + (np.mean(stream[zoom_pos:zoom_pos+template_len]) - np.mean(template_scaled))
        
        axs[2].plot(template_x, template_shifted, 'g-', label='Template')
        
        # Mark online and traditional detections near this position
        for pos in online_matches:
            if start <= pos <= end:
                axs[2].axvline(x=pos, color='r', linestyle='--', alpha=0.6)
        
        for pos in trad_matches:
            if start <= pos <= end:
                axs[2].axvline(x=pos, color='m', linestyle=':', alpha=0.6)
        
        axs[2].set_title(f'Detailed View of Match at Position {zoom_pos}')
        axs[2].legend()
    else:
        # If no matches available, show template alone
        axs[2].plot(template, 'g-', label='Template Pattern')
        axs[2].set_title('Template Pattern')
        axs[2].legend()
    
    axs[2].set_xlabel('Position')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('template_matching_results.png')
    plt.show()


def statistical_analysis(online_results, trad_results, online_scores, trad_scores, online_thresholds):
    """
    Perform advanced statistical analysis on the results with improved visualizations.
    
    Parameters:
    -----------
    online_results : dict
        Performance metrics for online method
    trad_results : dict
        Performance metrics for traditional method
    online_scores : list
        Similarity scores from online method
    trad_scores : list
        Similarity scores from traditional method
    online_thresholds : list
        Threshold values from online method
        
    Returns:
    --------
    dict : Statistical analysis results
    """
    # Create figure for statistical analysis
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Statistical Analysis of Template Matching Methods', fontsize=16)
    
    # 1. Score distributions with KDE
    from scipy.stats import gaussian_kde
    
    # Better visualization with kernel density estimation
    if len(online_scores) > 0:
        online_density = gaussian_kde(online_scores)
        trad_density = gaussian_kde(trad_scores)
        
        xs = np.linspace(min(min(online_scores), min(trad_scores)), 
                         max(max(online_scores), max(trad_scores)), 
                         1000)
        
        axs[0, 0].plot(xs, online_density(xs), 'r-', label='Online Method')
        axs[0, 0].plot(xs, trad_density(xs), 'b-', label='Traditional Method')
        
        # Histogram with transparency
        axs[0, 0].hist(online_scores, bins=30, alpha=0.3, density=True, color='r')
        axs[0, 0].hist(trad_scores, bins=30, alpha=0.3, density=True, color='b')
    else:
        axs[0, 0].text(0.5, 0.5, "Insufficient data for KDE", 
                      ha='center', va='center', transform=axs[0, 0].transAxes)
    
    # Add vertical lines for thresholds
    valid_thresholds = [t for t in online_thresholds if t is not None]
    if valid_thresholds:
        axs[0, 0].axvline(x=np.mean(valid_thresholds), color='r', linestyle='--', 
                         label='Avg Online Threshold')
    
    trad_threshold = 0.65  # Adjusted threshold
    axs[0, 0].axvline(x=trad_threshold, color='b', linestyle='--', 
                     label='Fixed Threshold')
    
    axs[0, 0].set_title('Similarity Score Distributions')
    axs[0, 0].set_xlabel('Similarity Score')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].legend()
    
    # 2. Performance metrics comparison
    metrics = ['Precision', 'Recall', 'F1 Score']
    online_metrics = [online_results[m] for m in metrics]
    trad_metrics = [trad_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axs[0, 1].bar(x - width/2, online_metrics, width, label='Online Method', color='r', alpha=0.7)
    axs[0, 1].bar(x + width/2, trad_metrics, width, label='Traditional Method', color='b', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(online_metrics):
        axs[0, 1].text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    for i, v in enumerate(trad_metrics):
        axs[0, 1].text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    axs[0, 1].set_title('Performance Metrics Comparison')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(metrics)
    axs[0, 1].set_ylabel('Score')
    axs[0, 1].set_ylim(0, 1.1)
    axs[0, 1].legend()
    
    # 3. ROC-like analysis (actually PR curve) if we have true matches
    true_indices = online_results.get('True Indices', [])
    if true_indices:
        # Create binary true labels (1 for true match locations, 0 elsewhere)
        binary_true = np.zeros(len(online_scores))
        for pos in true_indices:
            window_size = online_results.get('Window Size', 100)
            start = max(0, pos - window_size // 2)
            end = min(len(binary_true), pos + window_size // 2)
            binary_true[start:end] = 1
        
        if np.sum(binary_true) > 0:
            precision, recall, thresholds = precision_recall_curve(
                binary_true,
                np.array(online_scores)
            )
            
            pr_auc = auc(recall, precision)
            axs[1, 0].plot(recall, precision, 'r-', label=f'PR curve (AUC = {pr_auc:.3f})')
            axs[1, 0].set_title('Precision-Recall Curve')
            axs[1, 0].set_xlabel('Recall')
            axs[1, 0].set_ylabel('Precision')
            axs[1, 0].set_xlim(0, 1)
            axs[1, 0].set_ylim(0, 1)
            axs[1, 0].plot([0, 1], [0.5, 0.5], 'k--', alpha=0.3)
            axs[1, 0].plot([0.5, 0.5], [0, 1], 'k--', alpha=0.3)
            axs[1, 0].legend()
        else:
            axs[1, 0].text(0.5, 0.5, "Insufficient true matches for PR curve", 
                          ha='center', va='center', transform=axs[1, 0].transAxes)
    else:
        axs[1, 0].text(0.5, 0.5, "No true matches detected", 
                      ha='center', va='center', transform=axs[1, 0].transAxes)
    
    # 4. Online threshold evolution
    valid_positions = [(i, t) for i, t in enumerate(online_thresholds) if t is not None]
    if valid_positions:
        positions, thresholds = zip(*valid_positions)
        axs[1, 1].plot(positions, thresholds, 'g-', label='Dynamic Threshold')
        
        # Add trend line with confidence interval
        z = np.polyfit(positions, thresholds, 1)
        p = np.poly1d(z)
        axs[1, 1].plot(positions, p(positions), 'r--', 
                      label=f'Trend: {z[0]:.6f}x + {z[1]:.4f}')
        
        # Add confidence interval
        from scipy import stats
        
        # Calculate confidence interval
        n = len(positions)
        x_mean = np.mean(positions)
        y_mean = np.mean(thresholds)
        
        # Sum of squares
        SS_xx = sum((x - x_mean)**2 for x in positions)
        SS_xy = sum((x - x_mean)*(y - y_mean) for x, y in zip(positions, thresholds))
        SS_yy = sum((y - y_mean)**2 for y in thresholds)
        
        # Standard error of estimate
        SE = np.sqrt((SS_yy - SS_xy**2/SS_xx)/(n-2))
        
        # Confidence bands
        x_range = np.array(positions)
        y_fit = p(x_range)
        
        # 95% confidence interval
        t_value = stats.t.ppf(0.975, n-2)
        ci = t_value * SE * np.sqrt(1/n + (x_range - x_mean)**2/SS_xx)
        
        axs[1, 1].fill_between(x_range, y_fit-ci, y_fit+ci, color='r', alpha=0.2)
        
        axs[1, 1].set_title('Threshold Evolution with Trend Analysis')
        axs[1, 1].set_xlabel('Position in Stream')
        axs[1, 1].set_ylabel('Threshold Value')
        axs[1, 1].legend()
    else:
        axs[1, 1].text(0.5, 0.5, "No threshold evolution data available", 
                      ha='center', va='center', transform=axs[1, 1].transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('statistical_analysis.png')
    plt.show()
    
    # Calculate statistical significance between methods
    # Paired t-test of scores at match positions
    true_indices_set = set(online_results.get('True Indices', []))
    if true_indices_set:
        online_match_scores = [online_scores[i] for i in true_indices_set if i < len(online_scores)]
        trad_match_scores = [trad_scores[i] for i in true_indices_set if i < len(trad_scores)]
        
        # Ensure equal length and sufficient data
        min_len = min(len(online_match_scores), len(trad_match_scores))
        if min_len > 1:
            t_stat, p_value = stats.ttest_rel(online_match_scores[:min_len], trad_match_scores[:min_len])
        else:
            t_stat, p_value = 0, 1.0
    else:
        t_stat, p_value = 0, 1.0
    
    # Improvement calculation
    if trad_results['F1 Score'] > 0:
        f1_improvement = ((online_results['F1 Score'] - trad_results['F1 Score']) / 
                          max(trad_results['F1 Score'], 1e-10)) * 100
    else:
        if online_results['F1 Score'] > 0:
            f1_improvement = float('inf')  # Infinite improvement from zero
        else:
            f1_improvement = 0.0  # Both are zero
    
    # Additional insights
    mean_online = np.mean(online_scores)
    std_online = np.std(online_scores)
    mean_trad = np.mean(trad_scores)
    std_trad = np.std(trad_scores)
    
    # Separation of scores at true matches vs. other positions
    separation_analysis = {}
    if true_indices_set:
        # Scores at true match positions
        true_match_online_scores = [online_scores[i] for i in true_indices_set if i < len(online_scores)]
        other_online_scores = [s for i, s in enumerate(online_scores) if i not in true_indices_set]
        
        if true_match_online_scores and other_online_scores:
            # Calculate separation (effect size)
            true_mean = np.mean(true_match_online_scores)
            other_mean = np.mean(other_online_scores)
            pooled_std = np.sqrt((np.var(true_match_online_scores) + np.var(other_online_scores)) / 2)
            
            separation_analysis = {
                "True Match Mean Score": true_mean,
                "Non-Match Mean Score": other_mean,
                "Score Difference": true_mean - other_mean,
                "Effect Size (Cohen's d)": (true_mean - other_mean) / pooled_std,
                "Statistical Significance": stats.ttest_ind(true_match_online_scores, other_online_scores)
            }
    
    analysis_results = {
        "Method Comparison p-value": p_value,
        "F1 Score Improvement": f1_improvement,
        "Online Method Statistics": {
            "Mean Score": mean_online,
            "Std Deviation": std_online,
            "Signal-to-Noise Ratio": mean_online / std_online if std_online > 0 else 0
        },
        "Traditional Method Statistics": {
            "Mean Score": mean_trad,
            "Std Deviation": std_trad,
            "Signal-to-Noise Ratio": mean_trad / std_trad if std_trad > 0 else 0
        },
        "Online Dynamic Threshold": {
            "Mean": np.mean(valid_thresholds) if valid_thresholds else None,
            "Trend Slope": z[0] if 'z' in locals() else None,
            "Trend Confidence": ci[0] if 'ci' in locals() and len(ci) > 0 else None
        },
        "Separation Analysis": separation_analysis
    }
    
    return analysis_results


def create_matching_animation(stream, template, online_matches, true_matches, window_size=500, online_scores=None):
    """
    Create an animated visualization of the template matching process.
    
    Parameters:
    -----------
    stream : array-like
        The full data stream
    template : array-like
        The template pattern
    online_matches : list
        Positions where matches were found
    true_matches : list
        Positions of true matches
    window_size : int
        Size of the sliding window to show
    online_scores : list
        Similarity scores from online method
    """
    import matplotlib.animation as animation
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Template Matching Animation', fontsize=16)
    
    # Initialize plots
    stream_line, = ax1.plot([], [], 'b-', label='Stream')
    window_line, = ax1.plot([], [], 'r-', label='Current Window')
    template_line, = ax2.plot(template, 'g-', label='Template')
    
    # Set axis limits
    ax1.set_xlim(0, len(stream))
    ax1.set_ylim(min(stream) - 0.5, max(stream) + 0.5)
    ax2.set_xlim(0, len(template))
    ax2.set_ylim(min(template) - 0.5, max(template) + 0.5)
    
    # Add legends
    ax1.legend()
    ax2.legend()
    
    # Mark true matches with vertical lines
    for pos in true_matches:
        ax1.axvline(x=pos, color='g', alpha=0.3, linestyle='--')
    
    def init():
        stream_line.set_data([], [])
        window_line.set_data([], [])
        return stream_line, window_line
    
    def animate(frame):
        # Update stream view
        start = max(0, frame - window_size//2)
        end = min(len(stream), frame + window_size//2)
        stream_line.set_data(range(len(stream)), stream)
        
        # Update window view
        if frame < len(stream) - len(template):
            window_data = stream[frame:frame + len(template)]
            window_line.set_data(range(frame, frame + len(template)), window_data)
        
        # Add title showing match status
        if frame in online_matches:
            ax1.set_title(f'Frame {frame}: Match Found!', color='red')
        else:
            ax1.set_title(f'Frame {frame}')
        
        if online_scores is not None:
            ax1.set_ylabel(f'Score: {online_scores[frame]:.2f}')
        
        return stream_line, window_line
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=range(0, len(stream) - len(template), 50),
                                 interval=100, blit=False)
    
    # Save animation
    anim.save('template_matching.gif', writer='pillow')
    plt.close()


def run_experiment(stream, template, true_matches, observation_ratio=0.368, 
                 adaptation_rate=0.1, confidence_level=1.5, traditional_threshold=0.65):
    """
    Run an experiment comparing online and traditional template matching.
    
    Parameters:
    -----------
    stream : array-like
        Data stream
    template : array-like
        Template pattern
    true_matches : list
        Positions of true matches
    observation_ratio : float
        Ratio of the stream to observe before making decisions
    adaptation_rate : float
        Rate at which threshold adapts over time
    confidence_level : float
        Number of standard deviations for confidence bounds
    traditional_threshold : float
        Fixed threshold for traditional method
        
    Returns:
    --------
    dict : Experiment results
    """
    print("\n==== Running Template Matching Experiment ====")
    print(f"Stream length: {len(stream)}")
    print(f"Template length: {len(template)}")
    print(f"True matches: {len(true_matches)}")
    print(f"Online parameters: observation_ratio={observation_ratio}, " +
          f"adaptation_rate={adaptation_rate}, confidence_level={confidence_level}")
    print(f"Traditional threshold: {traditional_threshold}")
    
    # Initialize online matcher
    online_matcher = OnlineTemplateMatching(
        template, 
        observation_ratio=observation_ratio,
        adaptation_rate=adaptation_rate,
        confidence_level=confidence_level
    )
    
    # Process stream with online method
    print("\nProcessing with online method...")
    online_matches = online_matcher.process_stream(stream)
    
    # Process stream with traditional method
    print("\nProcessing with traditional method...")
    trad_matches, trad_scores = traditional_template_matching(stream, template, traditional_threshold)
    
    # Evaluate performance with larger window size for UCR data
    print("\nEvaluating performance...")
    window_size = min(len(template) // 2, 100)  # Use half of template length or 100, whichever is smaller
    online_results = evaluate_performance(online_matches, true_matches, len(stream), window_size)
    trad_results = evaluate_performance(trad_matches, true_matches, len(stream), window_size)
    
    # Print results
    print("\n==== Results ====")
    print("\nOnline Template Matching:")
    for k, v in online_results.items():
        if k != 'True Indices':  # Skip printing all indices
            print(f"  {k}: {v}")
    
    print("\nTraditional Template Matching:")
    for k, v in trad_results.items():
        if k != 'True Indices':  # Skip printing all indices
            print(f"  {k}: {v}")
    
    # Calculate improvement
    if trad_results['F1 Score'] > 0:
        f1_improvement = ((online_results['F1 Score'] - trad_results['F1 Score']) / 
                          max(trad_results['F1 Score'], 1e-10)) * 100
        print(f"\nF1 Score Improvement: {f1_improvement:.2f}%")
    else:
        if online_results['F1 Score'] > 0:
            print(f"\nF1 Score Improvement: Infinite (from 0 to {online_results['F1 Score']:.4f})")
        else:
            print("\nF1 Score Improvement: 0% (both methods failed)")
    
    # Statistical significance
    if online_results['p-value'] < 0.05:
        print("\nOnline method is statistically significant (p < 0.05)")
    else:
        print("\nOnline method is not statistically significant (p >= 0.05)")
    
    # Visualize results
    visualize_results(
        stream, 
        template, 
        online_matches,
        trad_matches,
        true_matches,
        online_matcher.similarity_scores,
        trad_scores,
        online_matcher.threshold_values,
        "Online vs Traditional Template Matching"
    )
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    analysis_results = statistical_analysis(
        online_results, 
        trad_results, 
        online_matcher.similarity_scores,
        trad_scores,
        online_matcher.threshold_values
    )
    
    # Create animation
    print("\nCreating animation of template matching process...")
    try:
        create_matching_animation(stream, template, online_matches, true_matches, online_scores=online_matcher.similarity_scores)
        print("Animation saved as 'template_matching.gif'")
    except Exception as e:
        print(f"Failed to create animation: {e}")
        print("Try installing pillow if not already installed: pip install pillow")
    
    # Print analysis results
    print("\n==== Statistical Analysis ====")
    for k, v in analysis_results.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for sk, sv in v.items():
                print(f"  {sk}: {sv}")
        else:
            print(f"{k}: {v}")
    
    # Return comprehensive results
    return {
        "Online Results": online_results,
        "Traditional Results": trad_results,
        "Online Matches": online_matches,
        "Traditional Matches": trad_matches,
        "Statistical Analysis": analysis_results,
        "Parameters": {
            "Observation Ratio": observation_ratio,
            "Adaptation Rate": adaptation_rate,
            "Confidence Level": confidence_level,
            "Traditional Threshold": traditional_threshold
        }
    }


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Online Template Matching with Optimal Stopping')
    parser.add_argument('--dataset', type=str, default="FordA", help='UCR dataset name (default: FordA)')
    parser.add_argument('--obs_ratio', type=float, default=0.368, help='Observation ratio (default: 0.368)')
    parser.add_argument('--adapt_rate', type=float, default=0.05, help='Adaptation rate (default: 0.05)')
    parser.add_argument('--conf_level', type=float, default=1.5, help='Confidence level (default: 1.5)')
    parser.add_argument('--trad_threshold', type=float, default=0.65, help='Traditional threshold (default: 0.65)')
    
    args = parser.parse_args()
    
    # Get data
    print(f"Using UCR dataset: {args.dataset}")
    stream, template, true_matches = prepare_ucr_data_for_template_matching(args.dataset)
    
    # Run experiment
    results = run_experiment(
        stream=stream,
        template=template,
        true_matches=true_matches,
        observation_ratio=args.obs_ratio,
        adaptation_rate=args.adapt_rate,
        confidence_level=args.conf_level,
        traditional_threshold=args.trad_threshold
    )
    
    print("\n==== Experiment Complete ====")
    print(f"Online method F1 Score: {results['Online Results']['F1 Score']:.4f}")
    print(f"Traditional method F1 Score: {results['Traditional Results']['F1 Score']:.4f}")
    
    # Final conclusion
    online_f1 = results['Online Results']['F1 Score']
    trad_f1 = results['Traditional Results']['F1 Score']
    
    if online_f1 > trad_f1:
        percent_improvement = ((online_f1 - trad_f1) / max(trad_f1, 1e-10)) * 100
        print(f"Improvement: {percent_improvement:.2f}%")
        print("The secretary-problem inspired approach performs better than traditional fixed-threshold matching.")
    elif online_f1 < trad_f1:
        percent_decrease = ((trad_f1 - online_f1) / trad_f1) * 100
        print(f"Traditional method performs better by {percent_decrease:.2f}%")
        print("For this dataset, the traditional approach outperformed the optimal stopping approach.")
    else:
        print("Both methods performed equally.")
        
    print("Results saved as template_matching_results.png and statistical_analysis.png")