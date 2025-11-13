from datetime import datetime, timedelta
import json
import statistics
import os

class ChatManager:
    def __init__(self, chat_save_file="potato_chat.json") -> None:
        self.chat_savefile_name = chat_save_file
        # Try to load existing chat history on initialization
        if os.path.exists(self.chat_savefile_name):
            with open(self.chat_savefile_name) as f:
                self.chat_list = json.loads(f.read())
        else:
            self.chat_list = list()

    def analyze_chat_patterns(self, chat_list):
        if len(chat_list) < 2:
            return {
                'typical_gap': timedelta(minutes=30),
                'activity_window': timedelta(hours=6),
                'min_messages': 3
            }

        times = [datetime.strptime(chat["time"], '%d-%m-%y %H:%M:%S') for chat in chat_list]
        time_differences = [
            (times[i] - times[i-1]).total_seconds()
            for i in range(1, len(times))
        ]

        # Calculate typical gap using quartiles to avoid extreme outliers
        sorted_diffs = sorted(time_differences)
        q1 = sorted_diffs[len(sorted_diffs)//4]
        q3 = sorted_diffs[3*len(sorted_diffs)//4]
        typical_gap = statistics.median(sorted_diffs)

        # Use IQR to determine significant gaps
        iqr = q3 - q1
        significant_gap = min(
            q3 + 1.5 * iqr,  # Statistical outlier threshold
            timedelta(hours=2).total_seconds()  # Reasonable upper bound
        )

        # Analyze message density patterns
        total_duration = (times[-1] - times[0]).total_seconds()
        avg_msgs_per_hour = len(chat_list) / (total_duration / 3600)

        # Adapt activity window based on message density
        if avg_msgs_per_hour > 20:  # Very active chat
            activity_window = timedelta(hours=2)
        elif avg_msgs_per_hour > 5:  # Moderately active
            activity_window = timedelta(hours=6)
        else:  # Less active
            activity_window = timedelta(hours=12)

        # Adapt minimum messages threshold based on activity level
        min_messages = max(2, int(avg_msgs_per_hour / 2))

        return {
            'typical_gap': timedelta(seconds=significant_gap),
            'activity_window': activity_window,
            'min_messages': min_messages
        }

    def calculate_cutoff_time(self, chat_list):
        if len(chat_list) < 2:
            return timedelta(hours=6)

        # Get adaptive parameters
        params = self.analyze_chat_patterns(chat_list)

        times = [datetime.strptime(chat["time"], '%d-%m-%y %H:%M:%S') for chat in chat_list]
        time_differences = [
            (times[i] - times[i-1]).total_seconds()
            for i in range(1, len(times))
        ]

        # Find clusters of activity using adaptive parameters
        max_gap = params['typical_gap'].total_seconds()
        activity_window = params['activity_window']
        min_messages = params['min_messages']

        for i in reversed(range(len(time_differences))):
            if time_differences[i] > max_gap:
                # Found a significant gap, check for substantial activity before this gap
                messages_before_gap = len([t for t in times[:i]
                    if t >= times[i] - activity_window])

                if messages_before_gap >= min_messages:
                    return datetime.now() - times[i]

        # Default window if no clear cutoff point found
        return params['activity_window'] * 2

    def filter_old_messages(self):
        current_time = datetime.now()
        cutoff_delta = self.calculate_cutoff_time(self.chat_list)
        cutoff_time = current_time - cutoff_delta

        # Filter out old messages
        self.chat_list = [
            chat for chat in self.chat_list
            if datetime.strptime(chat["time"], '%d-%m-%y %H:%M:%S') >= cutoff_time
        ]

    def add_chat(self, author, message, images: list = None):
        chat_dict = {
            "time": datetime.now().strftime('%d-%m-%y %H:%M:%S'),
            "speaker": author,
            "message": message,
            "images": images
        }

        self.chat_list.append(chat_dict)
        # Sort chat_list by time
        self.chat_list.sort(key=lambda x: datetime.strptime(x["time"], '%d-%m-%y %H:%M:%S'))

        self.filter_old_messages()  # Filter before saving

        with open(self.chat_savefile_name, "w") as f:
            f.write(json.dumps(self.chat_list, indent=4))


    def get_chat_list(self):
        # No need to reload from file since we're maintaining the state in memory
        self.filter_old_messages()
        return self.chat_list

    def clear_chat(self):
        self.chat_list = list()
        # Also clear the file
        with open(self.chat_savefile_name, "w") as f:
            f.write(json.dumps(self.chat_list, indent=4))
