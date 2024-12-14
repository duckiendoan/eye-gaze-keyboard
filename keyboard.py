import cv2
import numpy as np
import pyautogui
import time
from PIL import ImageFont, ImageDraw, Image
from screeninfo import get_monitors
import multiprocessing

# keys = [
#     ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
#     ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
#     ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
#     ['SPACE', 'BACKSPACE', 'ENTER', 'CLEAR']
# ]
DWELL_TIME = 0.9
KEY_SPACING = 5
KEY_SPACING_VERTICAL = 5
PREDICTION_COUNT = 3
KEY_PADDING = 25

# Define keyboard layout
main_keys = [
    ['k', 'đ', 'g', 'm', 'u', 'y'],
    ['x', 's', 'n', 'h', 'a', 'd'],
    ['f', 'o', 'SPACE', 'r', 'q'],
    ['w', 'v', 'c', 'i', 't', 'b'],
    ['p', 'l', 'e', 'z', 'BACKSPACE']
]

extended_keys = {
    "a":  ([["", "", "", "", ''], 
            ["ặ", "à", "á", "ấ",'ẩ'], 
            ["ằ", "ã", "a", "ả", 'ẳ'], 
            ["ẫ", "ậ", "ạ", "â", 'ẵ'], 
            ["BACK", "ắ", "ầ", "ă", '']]),
    "e":  ([["", "", "", "", ''], 
            ["", "ế", "ê", "ệ",''], 
            ["", "ẹ", "e", "ể", ''], 
            ["", "é", "ẽ", "ề", ''], 
            ["BACK", "è", "ễ", "ẻ", '']]),
    "y":  ([["", "", "", "", ''], 
            ["", "", "ỹ", "",''], 
            ["", "ý", "y", "ỳ", ''], 
            ["", "", "ỷ", "ỵ", ''], 
            ["BACK", "", "", "", '']]),
    "i":  ([["", "", "", "", ''], 
            ["", "", "ị", "",''], 
            ["", "ì", "i", "í", ''], 
            ["", "", "ĩ", "ỉ", ''], 
            ["BACK", "", "", "", '']]),
    "o":  ([["", "", "", "", ''], 
            ["õ", "o", "ó", "ố",'ỡ'], 
            ["ồ", "ơ", "ô", "ờ", 'ỗ'], 
            ["ọ", "ợ", "ộ", "ớ", 'ỏ'], 
            ["BACK", "ở", "ò", "ổ", '']]),
    "u":  ([["", "", "", "", ''], 
            ["", "ư", "ú", "ủ",''], 
            ["", "ụ", "u", "ứ", ''], 
            ["", "ự", "ừ", "ữ", ''], 
            ["BACK", "ũ", "ù", "ử", '']]),
}

def draw_text(image, position, text, font, color):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    xmin, ymin, xmax, ymax = font.getbbox(text)
    text_x = position[0] - (xmax - xmin) // 2
    text_y = position[1] - (ymax - ymin) // 2
    draw.text((text_x, text_y), text, font=font, fill=color)
    return np.array(pil_image)

def hex_to_bgr(hex_color):
    # Remove the '#' if it is present
    hex_color = hex_color.lstrip('#')
    
    # Check if the hex color is valid (length should be 6 or 3)
    if len(hex_color) not in (6, 3):
        raise ValueError("Invalid hex color format. Use 6 or 3 characters.")
    
    # If it's a shorthand hex (3 characters), expand it to 6 characters
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (b, g, r)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def predict(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        self._dfs(node, prefix, results)
        return results

    def _dfs(self, node, prefix, results):
        if node.is_end_of_word:
            results.append(prefix)
        for char, child in node.children.items():
            self._dfs(child, prefix + char, results)

class Keyboard:
    def __init__(self, keys, font, base_key_width=70, base_key_height=70, keyboard_width=1000, keyboard_height=700):
        self.keys = keys.copy()
        self.key_locs = [row.copy() for row in self.keys]
        self.key_sizes = [row.copy() for row in self.key_locs]
        self.font = font

        for row_index, row in enumerate(keys):
            for col_index, key in enumerate(row):
                x1, y1, x2, y2 = font.getbbox(key)
                self.key_sizes[row_index][col_index] = int(x2 - x1), int(y2 - y1)
        # print(self.key_sizes)

        self.base_key_width = base_key_width
        self.base_key_height = base_key_height
        self.keyboard_height = keyboard_height
        self.keyboard_width = keyboard_width
        self.text_color = (255, 255, 255)
        self.background_color = hex_to_bgr("#212121")
        self.border_key_color = hex_to_bgr("#424242")
        self.highlight_key_color = hex_to_bgr("#EEEEEE")
        self.focus_color = hex_to_bgr("#EEEEEE")
        self.col_spans = {
            'SPACE': 2,
            'BACKSPACE': 2
        }
        self.base_keyboard = self._draw_keys(keys, self.key_sizes)
        # State
        self.progress_states = {}
        self.predicted_words = []
    
    def _draw_keys(self, keys, key_sizes):
        # keyboard = np.zeros((self.keyboard_height, self.keyboard_width, 3), dtype=np.uint8)
        keyboard = np.full((self.keyboard_height, self.keyboard_width, 3), fill_value=self.background_color, dtype=np.uint8)
        img_pil = Image.fromarray(keyboard)
        draw = ImageDraw.Draw(img_pil)

        for row_index, row in enumerate(keys):
            row_key_widths = [max(self.base_key_width, w + KEY_PADDING * 2) for w, h in key_sizes[row_index]]
            row_key_widths = [x if row[i] not in self.col_spans else self.col_spans[row[i]] * (self.base_key_width + KEY_SPACING) - KEY_SPACING for i, x in enumerate(row_key_widths)]
            # row_width = len(row) * (key_width + KEY_SPACING) - KEY_SPACING
            row_width = sum(row_key_widths) + (len(row) - 1) * KEY_SPACING
            row_start_x = (self.keyboard_width - row_width) // 2  # Center each row individually
            accumulated_width = 0
            for col_index, key in enumerate(row):
                key_width = row_key_widths[col_index]
                key_height = self.base_key_height
                x = row_start_x if col_index == 0 else \
                        row_start_x + accumulated_width + col_index * KEY_SPACING
                y = row_index * (key_height + KEY_SPACING_VERTICAL) + 0.1 * self.keyboard_height + int(self.base_key_height) + 2 * KEY_SPACING_VERTICAL
                accumulated_width += key_width
                text_size = key_sizes[row_index][col_index]
                # Draw text
                text_x = x + (key_width - text_size[0]) // 2
                text_y = y + (key_height - text_size[1]) // 2
                if key:
                    draw.rectangle(((x, y), (x + key_width, y + key_height)), fill=self.border_key_color)
                    draw.text((text_x, text_y), key, font=self.font, fill = (*self.text_color, 1))
                
                # cv2.putText(keyboard, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return np.array(img_pil)

    def draw(self, highlighted_key, input_text, cursor_visible=True, mouse_x=None, mouse_y=None):
        # keyboard = np.zeros((keyboard_height, keyboard_width, 3), dtype=np.uint8)
        keyboard = Image.fromarray(self.base_keyboard)
        draw = ImageDraw.Draw(keyboard)
        # total_width = len(keys[0]) * (key_width + KEY_SPACING) - KEY_SPACING  # Adjusted for horizontal alignment
        # start_x = (keyboard_width - total_width) // 2  # Center alignment horizontally
        if mouse_x and mouse_y:
            draw.circle((mouse_x, mouse_y), 0.1 * self.base_key_height, fill=self.focus_color)
        # Draw predictive text buttons
        pred_button_width = (len(self.keys[0]) * (self.base_key_width + KEY_SPACING) - KEY_SPACING) // 3 - KEY_SPACING
        pred_start_x = (self.keyboard_width - (3 * pred_button_width + 2 * KEY_SPACING)) // 2
        text_with_cursor = input_text
        if cursor_visible:
            text_with_cursor += '|'
        draw.text((pred_start_x, 0.05 * self.keyboard_height), text_with_cursor, font=self.font, fill=self.text_color)

        for i, word in enumerate(self.predicted_words[:3]):
            x = pred_start_x + i * (pred_button_width + 10)  # Spaced above the keyboard
            y = 0.1 * self.keyboard_height
            draw.rectangle((x, y, x + pred_button_width, y + self.base_key_height), fill=self.border_key_color)
            if word in self.progress_states and self.progress_states[word] > 0:
                bar_height = int((self.progress_states[word] / DWELL_TIME) * self.base_key_height)  # Fill progress bar based on time
                draw.rectangle((x, y + self.base_key_height - bar_height, x + pred_button_width, y + self.base_key_height), fill=self.focus_color)
            # text_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            # text_x = x + (pred_button_width - text_size[0]) // 2
            # text_y = y + (self.base_key_height + text_size[1]) // 2
            text_position = (x + pred_button_width // 2, y + self.base_key_height // 2)
            xmin, ymin, xmax, ymax = self.font.getbbox(word)
            text_x = text_position[0] - (xmax - xmin) // 2
            text_y = text_position[1] - (ymax - ymin) // 2
            draw.text((text_x, text_y), word, font=self.font, fill=self.text_color)
            # keyboard = draw_text(keyboard, (x + pred_button_width // 2, y + self.base_key_height // 2), word, self.font, self.text_color)
            # cv2.putText(keyboard, word, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        for row_index, row in enumerate(self.keys):
            row_key_widths = [max(self.base_key_width, w + KEY_PADDING * 2) for w, h in self.key_sizes[row_index]]
            row_key_widths = [x if row[i] not in self.col_spans else self.col_spans[row[i]] * (self.base_key_width + KEY_SPACING) - KEY_SPACING for i, x in enumerate(row_key_widths)]
            row_width = sum(row_key_widths) + (len(row) - 1) * KEY_SPACING
            row_start_x = (self.keyboard_width - row_width) // 2  # Center each row individually
            accumulated_width = 0
            for col_index, key in enumerate(row):
                key_width = row_key_widths[col_index]
                key_height = self.base_key_height
                x = row_start_x if col_index == 0 else \
                        row_start_x + accumulated_width + col_index * KEY_SPACING
                y = row_index * (key_height + KEY_SPACING_VERTICAL) + 0.1 * self.keyboard_height + int(self.base_key_height) + 2 * KEY_SPACING_VERTICAL
                accumulated_width += key_width
                text_size = self.key_sizes[row_index][col_index]
                
                if key != '':
                    # Draw progress bar for the highlighted key
                    if key in self.progress_states and self.progress_states[key] > 0:
                        bar_height = int((self.progress_states[key] / DWELL_TIME) * key_height)  # Fill progress bar based on time
                        draw.rectangle((x, y + key_height - bar_height, x + key_width, y + key_height), fill=self.focus_color)
                        text_x = x + (key_width - text_size[0]) // 2
                        text_y = y + (key_height - text_size[1]) // 2
                        color = hex_to_bgr("#212121") if self.progress_states[key] / DWELL_TIME > 0.6 else self.text_color
                        draw.text((text_x, text_y), key, font=self.font, fill = (*color, 1))

                    # color = self.highlight_key_color if highlighted_key == key else self.border_key_color  # Highlight or white
                    # cv2.rectangle(keyboard, (x, y), (x + key_width, y + key_height), color, -1)
                self.key_locs[row_index][col_index] = (x, y, x + key_width, y + key_height)
                # cv2.putText(keyboard, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return np.array(keyboard)

    def simulate_gaze(self, mouse_x, mouse_y):
        # Check predictive buttons
        pred_button_width = (len(self.keys[0]) * (self.base_key_width + 10) - 10) // 3 - 10
        pred_start_x = (self.keyboard_width - (3 * pred_button_width + 20)) // 2
        for i, word in enumerate(self.predicted_words[:3]):
            x = pred_start_x + i * (pred_button_width + 10)
            y = 0.1 * self.keyboard_height
            if x <= mouse_x <= x + pred_button_width and y <= mouse_y <= y + self.base_key_height:
                # perform_action(word)
                return word

        for row_index, row in enumerate(self.keys):
            for col_index, key in enumerate(row):
                x1, y1, x2, y2 = self.key_locs[row_index][col_index]
                if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                    return key
        return None
    

# Initialize the Trie with words from a dictionary file
dictionary_file = "Viet11K.txt"
trie = Trie()
with open(dictionary_file, "r", encoding="utf8") as file:
    for line in file:
        trie.insert(line.strip().lower())

def perform_action(input_text, keyboard: Keyboard, key):
    if key == 'SPACE':
        input_text += ' '
    elif key == 'BACKSPACE':
        input_text = input_text[:-1]
    elif key == 'CLEAR':
        input_text = ""
    elif key == 'ENTER':
        print(f"Final Input: {input_text}")
        input_text = ""
    else:
        if key not in keyboard.predicted_words:
            input_text += key
        else:
            # Avoid duplicating characters when appending a predictive word
            overlap = 0
            for i in range(len(input_text)):
                if input_text[i:] == key[:len(input_text) - i]:
                    overlap = len(input_text) - i
                    break
            input_text = input_text[:len(input_text) - overlap] + key

    update_prediction(input_text, keyboard)   
    print(f"Input: {input_text}")

    return input_text
    # update_input_window()

def update_prediction(input_text, keyboard: Keyboard):
    current_word = input_text.split()[-1] if input_text.strip() else ""
    keyboard.predicted_words = trie.predict(current_word)

# Dummy predictive typing function
def predict_words(prefix):
    words = ["HELLO", "HEY", "HELP", "HAPPY", "HEART"]  # Example word set
    return [word for word in words if word.startswith(prefix.upper())]
    
def run_keyboard_process(mouse_coords):
    fontpath = "Roboto-Bold.ttf"     
    font = ImageFont.truetype(fontpath, 50)

    # base_keyboard = draw_keys(keys, key_sizes)
    # Initialize OpenCV windows
    cv2.namedWindow("Eye Gaze Controlled Keyboard")

    input_text = ""
    cursor_visible = True
    cursor_blink_time = 0.5  # Cursor blink interval in seconds
    last_cursor_toggle = time.time()

    running = True
    highlighted_key = None
    last_time = time.time()
    monitor = get_monitors()[0]
    
    max_len_keyboard = max(len(row) for row in main_keys)
    base_key_width = int(0.9 * (monitor.width - max_len_keyboard * KEY_SPACING) / (max_len_keyboard))
    base_key_height = int(0.6 * (monitor.height - len(main_keys) * KEY_SPACING_VERTICAL) / (len(main_keys)))
    main_keyboard = Keyboard(main_keys, font, base_key_width=base_key_width, base_key_height=base_key_height,
                             keyboard_width=monitor.width, keyboard_height=monitor.height)
    extended_keyboards = {}
    for k, ex_keys in extended_keys.items():
        extended_keyboards[k] = Keyboard(ex_keys, font, base_key_width=base_key_width, base_key_height=base_key_height,
                             keyboard_width=monitor.width, keyboard_height=monitor.height)

    current_keyboard = main_keyboard
    top_level = True

    while running:
        current_time = time.time()
        elapsed = current_time - last_time
        last_time = current_time

        # Get actual mouse position on screen
        # mouse_x, mouse_y = pyautogui.position()
        mouse_x, mouse_y = mouse_coords[:]
        # Convert mouse coordinates to the window coordinates
        win_x, win_y, _, _ = cv2.getWindowImageRect("Eye Gaze Controlled Keyboard")
        mouse_x -= win_x
        mouse_y -= win_y
        # if len(input_text) > 0:
        #     current_word = input_text.split()[-1] if input_text.strip() else ""
        #     current_keyboard.predicted_words = trie.predict(current_word)
        cv2.imshow("Eye Gaze Controlled Keyboard", current_keyboard.draw(highlighted_key, input_text, cursor_visible, mouse_x, mouse_y))
        
        if time.time() - last_cursor_toggle >= cursor_blink_time:
            cursor_visible = not cursor_visible
            last_cursor_toggle = time.time()

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # Exit on ESC key
            running = False

        # Get actual mouse position on screen
        # mouse_x, mouse_y = pyautogui.position()
        # # Convert mouse coordinates to the window coordinates
        # win_x, win_y, _, _ = cv2.getWindowImageRect("Eye Gaze Controlled Keyboard")
        # mouse_x -= win_x
        # mouse_y -= win_y

        new_highlighted_key = current_keyboard.simulate_gaze(mouse_x, mouse_y)

        if new_highlighted_key != highlighted_key:
            # Reset progress of the previously highlighted key
            if highlighted_key:
                current_keyboard.progress_states[highlighted_key] = 0
            highlighted_key = new_highlighted_key
            if highlighted_key:
                current_keyboard.progress_states[highlighted_key] = 0  # Reset progress for the newly highlighted key

        # Update progress for the currently highlighted key
        if highlighted_key:
            current_keyboard.progress_states[highlighted_key] = current_keyboard.progress_states.get(highlighted_key, 0) + elapsed
            if current_keyboard.progress_states[highlighted_key] >= DWELL_TIME:
                if highlighted_key in extended_keys and top_level:
                    current_keyboard.progress_states[highlighted_key] = 0
                    current_keyboard = extended_keyboards[highlighted_key]
                    top_level = False
                    update_prediction(input_text, current_keyboard)
                else:
                    if highlighted_key != 'BACK':
                        input_text = perform_action(input_text, current_keyboard, highlighted_key)
                    current_keyboard.progress_states[highlighted_key] = 0
                    if not top_level:
                        current_keyboard = main_keyboard
                        top_level = True
                        update_prediction(input_text, current_keyboard)
                

    cv2.destroyAllWindows()


def run_mouse_update(keyboard_proc, mouse_coords):
    try:
        while keyboard_proc.is_alive():
            mouse_x, mouse_y = pyautogui.position()
            mouse_coords[0], mouse_coords[1] = mouse_x, mouse_y
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if keyboard_proc.is_alive():
            keyboard_proc.terminate()
        keyboard_proc.join()

def main():
    mouse_coords = multiprocessing.Array('i', [0, 0])
    keyboard_proc = multiprocessing.Process(target=run_keyboard_process, args=(mouse_coords,))
    keyboard_proc.start()
    run_mouse_update(keyboard_proc,mouse_coords)

    keyboard_proc.join()

if __name__ == "__main__":
    main()
