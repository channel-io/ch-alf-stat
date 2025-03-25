import os
import json

from logging import getLogger   
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from src.model import AlfLog, AlfTurn
from pymongo import MongoClient
logger = getLogger(__name__)


class LogHandler:
    """
    LogHandler class is responsible for handling and processing data logs.

    Attributes:
        data_dir (str): The directory where data logs are stored.
        subdirs (Optional[List[str]]): List of subdirectories to include in the data loading process.
        start_date (Optional[str]): The start date for filtering logs. (Timestamp)
        end_date (Optional[str]): The end date for filtering logs. (Timestamp)
    """
    def __init__(
        self,
        logs_dir: Optional[str] = None,
        subdirs: Optional[List[str]] = None,
        channel_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        if not (channel_id or logs_dir):
            raise ValueError("Either one of channel_id or logs_dir must be provided")
        if channel_id and not start_date:
            raise ValueError("start_date must be provided if channel_id is provided")

        self.channel_id = channel_id
        self.logs_dir = logs_dir
        self.subdirs = subdirs
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        self.raw_logs = self._load_logs()
        self.chat_ids = self._get_chat_ids()
        self.logs = self.preprocess_logs()

    def _load_logs(self):
        if self.channel_id:
            return self._load_logs_from_mongodb()
        else:
            return self._load_logs_from_local()
    
    def _load_logs_from_mongodb(self):
        mongo_client = MongoClient(os.getenv("MONGO_URI"))
        db = mongo_client["channel"]["alf_logs"]
        logs = list(db.find({"channel_id": self.channel_id}))
        if self.start_date:
            logs = [log for log in logs if log["metadata"]["utc_time"] >= self.start_date]
        if self.end_date:
            logs = [log for log in logs if log["metadata"]["utc_time"] <= self.end_date]
        logs.sort(key=lambda x: datetime.strptime(x["metadata"]["utc_time"], "%Y-%m-%d %H:%M:%S.%f%z"))
        logger.info(f"Loaded {len(logs)} logs")
        return logs

    def _load_logs_from_local(self):
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

    def _get_chat_ids(self):
        """
        Get the chat_ids from the logs.
        """
        valid_data = []
        for d in self.raw_logs:
            if 'generate_answer_with_knowledge' in d:
                valid_data.append(d['generate_answer_with_knowledge'])
            elif 'generate_answer' in d:
                valid_data.append(d['generate_answer'])

        chat_ids = set()
        for d in valid_data:
            if 'request' in d and 'response' in d:
                chat_ids.add(d['request']['chat_id'])
        return list(chat_ids)
    

    def preprocess_logs(self) -> List[AlfLog]:
        """
        This function preprocesses the data by grouping it by chat_id and converting it to AlfLog objects with AlfTurn items.
        
        It iterates through the data logs and organizes them into a dataset dictionary where each key is a chat_id
        and each value is an AlfLog object.
        """
        raw_dataset = {}

        for item in self.raw_logs:
            chat_id = item["generate_answer"]["request"]["chat_id"]
            if chat_id not in raw_dataset:
                raw_dataset[chat_id] = {"turns": [], "feedback": None}
            
            uid = item['metadata']['uid']
            if "utc_time_in" in item['metadata']:
                time_in = item['metadata']['utc_time_in']
                time_out = item['metadata']['utc_time']
                response_latency = item["metadata"]["response_latency"]
                if isinstance(response_latency, str):
                    try:
                        latency_dt = datetime.strptime(response_latency, "%H:%M:%S.%f")
                        response_latency = latency_dt.hour * 3600 + latency_dt.minute * 60 + latency_dt.second + latency_dt.microsecond / 1000000
                    except ValueError:
                        logger.warning(f"Could not parse response_latency: {response_latency}")
                        response_latency = None
            else:
                time_in = item['metadata']['utc_time']
                time_out = None
                response_latency = None

            if "generate_answer_with_knowledge" in item:
                subitem = item["generate_answer_with_knowledge"]
                if "request" not in subitem or "response" not in subitem:
                    continue
                query = subitem["request"]["queries"][-1]
                if not query:
                    if subitem["request"]["messages"][-1]["plainText"]:
                        query = subitem["request"]["messages"][-1]["plainText"]
                    elif subitem["request"]["messages"][-1]["files"]:
                        query = "Image input"
                    else:
                        query = "Unknown input"
                summary = subitem["request"]["summary"]
                with_knowledge = True
                response_type = subitem["response"]["type"]
                response = subitem["response"]["message"]
                if "references" in subitem["response"]:
                    reference = [f"{r['type']}_{r['id']}" for r in subitem["response"]["references"]]
            
            else:
                subitem = item["generate_answer"]
                if "request" not in subitem or "response" not in subitem:
                    continue
                query = subitem["request"]["messages"][-1]["plainText"]
                with_knowledge = False
                response_type = subitem["response"]["type"]
                response = subitem["response"]["message"]
            
            alf_turn = AlfTurn(
                uid=uid,
                time_in=time_in,
                time_out=time_out,
                response_latency=response_latency,
                query=query,
                summary=summary,
                with_knowledge=with_knowledge,
                response_type=response_type,
                response=response,
                reference=reference,    
            )
            raw_dataset[chat_id]["turns"].append(alf_turn)

        # Sort turns by time_in and determine if they were sent to the user
        if not self.channel_id:
            for _, chat_data in raw_dataset.items():
                chat_data["turns"].sort(key=lambda x: x.time_in)
                for i, turn in enumerate(chat_data["turns"]):
                    turn.sent = all(turn.time_out <= next_turn.time_in for next_turn in chat_data["turns"][i+1:])

        # Convert to AlfLog and AlfTurn models
        processed_dataset = []
        for chat_id, chat_data in raw_dataset.items():
            channel_id = self.channel_id or "local"
            alf_log = AlfLog(
                channel_id=channel_id,
                chat_id=chat_id,
                turns=chat_data["turns"]
            )
            processed_dataset.append(alf_log)
            
        return processed_dataset


    def split_data_by_response_type(self):
        """
        Split the dataset into knowledge-based and function-call datasets.
        """
        turns_kb = []
        turns_fc = []
        
        for alf_log in self.logs:
            for turn in alf_log.turns:
                if turn.sent:  # Filter out responses that are not sent to user
                    if turn.with_knowledge:
                        turns_kb.append(turn)
                    else:
                        turns_fc.append(turn)
                        
        turns_kb.sort(key=lambda x: x.time_in)
        turns_fc.sort(key=lambda x: x.time_in)
        
        return turns_kb, turns_fc
    

    def get_unsatisfied_chats(self):
        """
        Unsatisfied chats are:
        - At least more than one knowledge-based response is provided.
        - The last response is open_chat.
        - Handed to human within 24 hours.
        """
        unsatisfied_chats = set()
        for chat_id, alf_log in self.dataset.items():
            turns = alf_log.turns
            if turns and turns[-1].response_type == 'open_chat':
                for turn in turns:
                    if turn.sent and turn.response_type in ['faq', 'rag']:
                        if datetime.strptime(turns[-1].time_in, "%Y-%m-%d %H:%M:%S.%f%z") - datetime.strptime(turn.time_in, "%Y-%m-%d %H:%M:%S.%f%z") < timedelta(hours=24):
                            unsatisfied_chats.add(chat_id)
                            break
                            
        unsatisfied_logs = {k: v for k, v in self.dataset.items() if k in unsatisfied_chats}
        logger.info(f"Number of unsatisfied chats: {len(unsatisfied_logs)}")
        return unsatisfied_logs
    