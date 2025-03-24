import os
import json

from logging import getLogger   
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Optional, List

logger = getLogger(__name__)

class LogHandler:
    """
    LogHandler class is responsible for handling and processing data logs.

    Attributes:
        data_dir (str): The directory where data logs are stored.
        subdirs (Optional[List[str]]): List of subdirectories to include in the data loading process.
        start_date (Optional[str]): The start date for filtering logs. (Timestamp)
        end_date (Optional[str]): The end date for filtering logs. (Timestamp)

    Terminology:
        - "log": Raw log data (includes all information acquired from loki)
        - "data": Preprocessed data (includes only necessary information, grouped by chat_id, consists of turns)
        - "turn": Preprocessed data (not grouped by chat_id)
    """
    def __init__(
        self,
        logs_dir: str,
        subdirs: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        logger.info(f"Loading logs from {logs_dir}")
        logger.info(f"Subdirs: {subdirs}")
        self.logs_dir = logs_dir
        self.subdirs = subdirs
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S.%f%z") if start_date else None
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S.%f%z") if end_date else None
        self.logs = self.load_logs()  # raw data
        self.chat_ids = self.get_chat_ids()
        self.dataset = self.preprocess_data()  # preprocessed data
        self.turns_kb, self.turns_fc = self.split_data_by_response_type()

    def load_logs(self):
        logs = []
        for root, _, files in os.walk(self.logs_dir):
            if self.subdirs is not None:
                if root.split('/')[-1] not in self.subdirs:
                    continue
            for file in tqdm(files, desc="Loading log files"):
                if file.endswith("logs"):
                    with open(os.path.join(root, file), "r") as f:
                        logs.extend(json.load(f))
        if self.start_date:
            logs = [log for log in logs if log["metadata"]["utc_time_in"] >= self.start_date]
        if self.end_date:
            logs = [log for log in logs if log["metadata"]["utc_time_in"] <= self.end_date]
        logs.sort(key=lambda x: datetime.strptime(x["metadata"]["utc_time_in"], "%Y-%m-%d %H:%M:%S.%f%z"))
        logger.info(f"Loaded {len(logs)} logs")
        return logs


    def get_chat_ids(self):
        """
        Get the chat_ids from the logs.
        """
        valid_data = []
        for d in self.logs:
            if 'generate_answer_with_knowledge' in d:
                valid_data.append(d['generate_answer_with_knowledge'])
            elif 'generate_answer' in d:
                valid_data.append(d['generate_answer'])

        chat_ids = set()
        for d in valid_data:
            if 'request' in d and 'response' in d:
                chat_ids.add(d['request']['chat_id'])
        return list(chat_ids)
    

    def preprocess_data(self):
        """
        This function preprocesses the data by grouping it by chat_id and retaining only the necessary values.
        
        It iterates through the data logs and organizes them into a dataset dictionary where each key is a chat_id.
        For each chat_id, it stores a list of turns (interactions) and feedback (initially set to None).
        
        Each turn contains:
        - time_in: The time the interaction started.
        - time_out: The time the interaction ended.
        - response_latency: The latency of the response.
        - query: The query made by the user.
        - summary: A summary of the interaction.
        - knowledge_based: A boolean indicating if the response was knowledge-based.
        - response_type: The type of response.
        - response_message: The message of the response.
        - reference: A list of references (if any).
        - uid: A unique identifier for the interaction.
        - sent: A boolean indicating if the response was sent before the next interaction.

        The function ensures that the turns for each chat_id are sorted by time_in and sets the 'sent' flag for each turn.
        """
        dataset = {}

        for item in self.logs:
            chat_id = item["generate_answer"]["request"]["chat_id"]
            if chat_id not in dataset:
                dataset[chat_id] = {"turns": [], "feedback": None}
            
            turn = {
                "time_in": item["metadata"]["utc_time_in"],
                "time_out": item["metadata"]["utc_time"],
                "response_latency": item["metadata"]["response_latency"],
            }

            if "generate_answer_with_knowledge" in item:
                subitem = item["generate_answer_with_knowledge"]
                if "request" not in subitem or "response" not in subitem:
                    continue
                turn["query"] = subitem["request"]["queries"][-1]
                if not turn["query"]:
                    if subitem["request"]["messages"][-1]["plainText"]:
                        turn["query"] = subitem["request"]["messages"][-1]["plainText"]
                    elif subitem["request"]["messages"][-1]["files"]:
                        turn["query"] = "Image input"
                    else:
                        turn["query"] = "Unknown input"
                turn["summary"] = subitem["request"]["summary"]
                turn["knowledge_based"] = True
                turn["response_type"] = subitem["response"]["type"]
                turn["response_message"] = subitem["response"]["message"]
                if "references" in subitem["response"]:
                    turn["reference"] = [f"{r['type']}_{r['id']}" for r in subitem["response"]["references"]]
            
            else:
                subitem = item["generate_answer"]
                if "request" not in subitem or "response" not in subitem:
                    continue
                turn["query"] = subitem["request"]["messages"][-1]["plainText"]
                turn["knowledge_based"] = False
                turn["response_type"] = subitem["response"]["type"]
                turn["response_message"] = subitem["response"]["message"]

            turn['uid'] = item['metadata']['uid']
            dataset[chat_id]["turns"].append(turn)
            dataset[chat_id]["final_response_type"] = subitem["response"]["type"]

        for chat_id, chat_data in dataset.items():
            chat_data["turns"].sort(key=lambda x: x["time_in"])
            for i, turn in enumerate(chat_data["turns"]):
                turn["sent"] = all(turn["time_out"] <= next_turn["time_in"] for next_turn in chat_data["turns"][i+1:])

        return dataset
    

    def get_final_response_type(self):
        """
        Get the final response type from the dataset.
        """
        final_chat_types = {}
        for chat_id, chat_data in self.dataset.items():
            if chat_data['turns'][-1]['response_type'] not in final_chat_types:
                final_chat_types[chat_data['turns'][-1]['response_type']] = 0
            final_chat_types[chat_data['turns'][-1]['response_type']] += 1
        return final_chat_types


    def split_data_by_response_type(self):
        """
        Split the dataset into knowledge-based and function-call datasets.
        """
        turns = []
        for _, chat_data in self.dataset.items():
            for turn in chat_data["turns"]:
                if turn["sent"]:  # Filter out responses that are not sent to user
                    turns.append(turn)
        turns = sorted(turns, key=lambda x: x["time_in"])
        dataset_kb = [t for t in turns if t["knowledge_based"]]
        dataset_fc = [t for t in turns if not t["knowledge_based"]]
        return dataset_kb, dataset_fc
    

    def get_unsatisfied_chats(self):
        """
        Unsatisfied chats are:
        - At least more than one knowledge-based response is provided.
        - The last response is open_chat.
        - Handed to human within 24 hours.
        """
        unsatisfied_chats = set()
        for chat_id, chat_data in self.dataset.items():
            if chat_data['turns'][-1]['response_type'] == 'open_chat':
                for turn in chat_data['turns']:
                    if turn['sent'] and turn['response_type'] in ['faq', 'rag']:
                        if datetime.strptime(chat_data['turns'][-1]['time_in'], "%Y-%m-%d %H:%M:%S.%f%z") - datetime.strptime(turn['time_in'], "%Y-%m-%d %H:%M:%S.%f%z") < timedelta(hours=24):
                            unsatisfied_chats.add(chat_id)
        unsatisfied_chats = {k: v for k, v in self.dataset.items() if k in unsatisfied_chats}
        logger.info(f"Number of unsatisfied chats: {len(unsatisfied_chats)}")
        return unsatisfied_chats
    