import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from PIL import Image

def load_data_from_folders(base_dir, img_size=(64, 64), threshold=0.3):
    label_names = ['0', '1', '3', '5', '6', '9']
    data_by_class = {label: [] for label in label_names}

    for label_name in label_names:
        label_path = os.path.join(base_dir, label_name)
        if not os.path.isdir(label_path):
            continue  # 폴더가 없으면 건너뜀

        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(label_path, filename)
                try:
                    # 이미지 로드 및 전처리
                    img = Image.open(img_path).convert("L")
                    img = img.resize(img_size)
                    img_array = np.array(img) / 255.0  # [0, 1] 범위로 정규화
                    binary_img = (img_array > threshold).astype(np.float32)  # 이진화

                    data_by_class[label_name].append(binary_img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return data_by_class

def train_pixel_probabilities(data_by_class):
    class_pixel_probs = {}
    class_counts = {}
    total_images = 0

    for label, images in data_by_class.items():
        if len(images) == 0:
            continue  # 해당 클래스에 이미지가 없으면 건너뜀
        # 이미지 스택 생성
        image_stack = np.stack(images)  # (N, H, W)
        # 픽셀별로 평균 계산 (라플라스 스무딩 적용 가능)
        pixel_prob = (np.sum(image_stack, axis=0) + 1) / (len(images) + 2)  # 라플라스 스무딩
        class_pixel_probs[label] = pixel_prob
        class_counts[label] = len(images)
        total_images += len(images)

    return class_pixel_probs, class_counts, total_images

def calculate_class_priors(class_counts, total_images):
    class_priors = {}
    for label, count in class_counts.items():
        class_priors[label] = count / total_images
    return class_priors

def load_map(file_path):
    """Load the binary map from a .npy file."""
    return np.load(file_path)

def display_map_with_rover(map_array, rover_position, visited_white, visited_black, ax):
    """Display the map with the rover's position and visited locations."""
    display_map = np.full_like(map_array, 0.5, dtype=np.float32)  # Default: unexplored (gray)

    # Update the map for visited tiles
    for r, c in visited_white:
        display_map[r, c] = 1.0  # White: visited and white tile
    for r, c in visited_black:
        display_map[r, c] = 0.0  # Black: visited and black tile

    # Display the map
    ax.clear()
    ax.imshow(display_map, cmap="gray", vmin=0, vmax=1)
    ax.scatter(rover_position[1], rover_position[0], c="red", label="Rover")  # Rover's position
    ax.set_title("Rover Exploration")
    ax.axis("off")

def calculate_posterior_probs(visited_white, visited_black, map_array, class_pixel_probs, class_probs_list, sorted_labels, smoothing=1e-2):
    observed = np.full_like(map_array, np.nan, dtype=np.float32)

    # Mark visited white areas as 1 and black areas as 0
    for r, c in visited_white:
        observed[r, c] = 1
    for r, c in visited_black:
        observed[r, c] = 0

    num_classes = len(sorted_labels)
    posterior_probs = np.zeros(num_classes)

    for idx, label in enumerate(sorted_labels):
        prob = class_pixel_probs[label]
        likelihood = np.nansum(
            observed * np.log(prob + smoothing) +
            (1 - observed) * np.log(1 - prob + smoothing)
        )
        posterior_probs[idx] = likelihood

    posterior_probs = np.exp(posterior_probs - np.max(posterior_probs))
    posterior_probs *= class_probs_list
    posterior_probs /= posterior_probs.sum()

    uniform_probs = np.ones(num_classes) / num_classes
    num_pixels_observed = len(visited_white) + len(visited_black)
    total_pixels = map_array.size

    observation_ratio = (num_pixels_observed / total_pixels) ** 0.5

    posterior_probs = observation_ratio * posterior_probs + (1 - observation_ratio) * uniform_probs
    posterior_probs /= posterior_probs.sum()

    return posterior_probs

def get_next_move_to_target(rover_position, target_position):
    rover_row, rover_col = rover_position
    target_row, target_col = target_position

    if rover_row < target_row:  # Move down
        return 1, 0
    elif rover_row > target_row:  # Move up
        return -1, 0
    elif rover_col < target_col:  # Move right
        return 0, 1
    elif rover_col > target_col:  # Move left
        return 0, -1

    return 0, 0  # Already at the target

def find_nearest_white_cell(map_array, rover_position, visited):
    white_cells = np.argwhere(map_array == 1)  # Find all white cells
    unvisited = [tuple(cell) for cell in white_cells if tuple(cell) not in visited]

    if not unvisited:
        return None

    distances = [np.abs(rover_position[0] - cell[0]) + np.abs(rover_position[1] - cell[1]) for cell in unvisited]
    nearest_cell = unvisited[np.argmin(distances)]

    return nearest_cell

class RoverSimulation:
    def __init__(self, map_array, start_position, class_pixel_probs, class_probs_list, sorted_labels, true_label):
        self.map_array = map_array
        self.rover_position = start_position
        self.visited_white = set()  # Tiles visited and white
        self.visited_black = set()  # Tiles visited and black
        self.target = None
        self.class_pixel_probs = class_pixel_probs
        self.class_probs_list = class_probs_list
        self.sorted_labels = sorted_labels
        self.true_label = true_label  # 정답 라벨 추가
        self.fig = None
        self.ax_map = None
        self.ax_bar = None
        self.anim = None
        self.step_count = 0  # 스텝 수 추적 변수 추가

    def update(self, frame):
        self.step_count += 1  # 스텝 수 증가
        print(f"Frame: {frame}, Step: {self.step_count}, Rover position: {self.rover_position}, Target: {self.target}")
        if self.target is None or self.rover_position == self.target:
            visited = self.visited_white.union(self.visited_black)
            self.target = find_nearest_white_cell(self.map_array, self.rover_position, visited)

            if self.target is None:
                print("All white areas explored.")
                self.anim.event_source.stop()  # Stop the animation
                return

        dr, dc = get_next_move_to_target(self.rover_position, self.target)
        self.rover_position = (self.rover_position[0] + dr, self.rover_position[1] + dc)

        if self.map_array[self.rover_position] == 1:
            self.visited_white.add(self.rover_position)  # Visited and white
        else:
            self.visited_black.add(self.rover_position)  # Visited and black

        if self.rover_position == self.target:
            self.target = None

        self.update_display()

    def run(self):
        self.fig, (self.ax_map, self.ax_bar) = plt.subplots(1, 2, figsize=(12, 6))
        self.anim = FuncAnimation(self.fig, self.update, interval=200, repeat=False)
        plt.show()

    def update_display(self):
        self.ax_map.clear()
        self.ax_bar.clear()

        display_map_with_rover(self.map_array, self.rover_position, self.visited_white, self.visited_black, self.ax_map)

        posterior_probs = calculate_posterior_probs(
            self.visited_white, self.visited_black, self.map_array, self.class_pixel_probs, self.class_probs_list, self.sorted_labels
        )

        self.ax_bar.bar(range(len(self.sorted_labels)), posterior_probs)
        self.ax_bar.set_title("Probability Distribution")
        self.ax_bar.set_xlabel("Labels")
        self.ax_bar.set_ylabel("Probability")
        self.ax_bar.set_xticks(range(len(self.sorted_labels)))
        self.ax_bar.set_xticklabels(self.sorted_labels)
        self.ax_bar.set_ylim(0, 1)

        # Left bottom corner에 스텝 수와 정답 라벨 표시
        step_info = f"Step: {self.step_count}"
        true_label_info = f"True Label: {self.true_label}"
        text_str = f"{step_info}\n{true_label_info}"
        self.ax_map.text(0.05, 0.95, text_str, transform=self.ax_map.transAxes,
                         fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.5))

# 데이터 로드 및 모델 학습
data_by_class = load_data_from_folders("./expanded", img_size=(64, 64), threshold=0.3)
class_pixel_probs, class_counts, total_images = train_pixel_probabilities(data_by_class)
class_priors = calculate_class_priors(class_counts, total_images)

# 필요한 변수 설정
sorted_labels = list(class_pixel_probs.keys())
num_classes = len(sorted_labels)
class_probs_list = [class_priors[label] for label in sorted_labels]

# Load map
binary_map = np.load("./test_map/label00.npy")
start_pos = (32, 32)  # Starting near the center of the map

# True label 설정 (예시로 '0'으로 설정)
true_label = '0'

# Run simulation
simulation = RoverSimulation(binary_map, start_pos, class_pixel_probs, class_probs_list, sorted_labels, true_label)
simulation.run()

