import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import random
HM_SIZE = 10 #Harmony Memory size(number of harmonies)
FEATURE_COUNT = 10000 #For Arcene dataset
ALPHA = 0.7 #Accuracy weight
BETA = 0.3 #Feature reduction weight
CV_FOLDS = 5 #Cross-validadation folds
accuracy_hs = []
feature_counts_hs = []

# Harmony memory initialization
def initialize_harmony_memory(hm_size = HM_SIZE, feature_count = FEATURE_COUNT):
    HM = np.random.randint(0, 2, size = (hm_size, feature_count))
    return HM

# Fitness Function
def evaluate_fitness(harmony, X, y, alpha = ALPHA, beta = BETA):
    # Selected features from where harmony == 1
    selected_indices = np.where(harmony == 1)[0]

    # Edge case: no features selected(penalize heavily)
    if len(selected_indices) == 0:
        return 1e6
    
    X_selected = X[:, selected_indices]

    # SVM classifier for evaluation
    clf = SVC(kernel = 'linear', C = 1)
    scores = cross_val_score(clf, X_selected, y, cv = CV_FOLDS, scoring = 'accuracy')
    accuracy = np.mean(scores)
    fitness = alpha * (1 - accuracy) + beta * (len(selected_indices) / FEATURE_COUNT)
    return fitness

# Improvise New Harmony
def improvise_new_harmony(HM, HMCR = 0.9, PAR = 0.3, BW = 0.05):
    harmony_length = HM.shape[1]
    new_harmony = np.zeros(harmony_length, dtype = int)

    for i in range(harmony_length):
        if random.random() < HMCR:
            # Memory consideration: pick from existing harmony
            value = HM[random.randint(0, HM.shape[0] - 1)][i]
        else:
            # randomization
            value = random.randint(0, 1)

        # pitch adjustment(with probability PAR)
        if random.random() < PAR:
            if random.random() < BW:
                value = 1 - value #flip bit

        new_harmony[i] = value
    return new_harmony

# Harmony memory update
def update_harmony_memory(HM, fitness_scores, new_harmony, new_fitness):
    # replace worst harmony if new one is better
    worst_index = np.argmax(fitness_scores)
    if new_fitness < fitness_scores[worst_index]:
        HM[worst_index] = new_harmony
        fitness_scores[worst_index] = new_fitness
        return True
    return False