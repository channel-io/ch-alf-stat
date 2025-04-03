import os
import json

from logging import getLogger   
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from src.model import ALFLog, ALFChat, ALFFactCheck, ALFFunctionCall
from pymongo import MongoClient
from src.utils import detect_language
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
        self.chats, self.logs = self.preprocess_logs()

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
    

    def preprocess_logs(self) -> List[ALFLog]:
        """
        This function preprocesses the data by grouping it by chat_id and converting it to ALFLog objects with ALFLog items.
        
        It iterates through the data logs and organizes them into a dataset dictionary where each key is a chat_id
        and each value is an ALFLog object.
        """
        raw_dataset = {}

        for item in self.raw_logs:
            chat_id = item["generate_answer"]["request"]["chat_id"]
            if chat_id not in raw_dataset:
                raw_dataset[chat_id] = {"logs": [], "feedback": None}
            
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

            fact_check = None
            references = None
                
            # KB chat
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
                    references = [f"{r['type']}_{r['id']}" for r in subitem["response"]["references"]]
                if "fact_check" in item:
                    fact_check = ALFFactCheck(
                        is_fact=item["fact_check"]["response"]["is_fact"],
                        critic=item["fact_check"]["response"]["critic"],
                        rubric=item["fact_check"]["response"]["rubric"]
                    )
                    
            # FC chat
            else:
                subitem = item["generate_answer"]
                if "request" not in subitem or "response" not in subitem:
                    continue
                query = subitem["request"]["messages"][-1]["plainText"]
                with_knowledge = False
                response_type = subitem["response"]["type"]
                response = subitem["response"]["message"]

            function_call = ALFFunctionCall(
                name=item["function call"]["name"],
                arguments=item["function call"]["arguments"]
            )
            
            alf_log = ALFLog(
                uid=uid,
                time_in=time_in,
                time_out=time_out,
                response_latency=response_latency,
                query=query,
                summary=summary,
                with_knowledge=with_knowledge,
                response_type=response_type,
                response=response,
                references=references,    
                function_call=function_call,
                fact_check=fact_check
            )
            raw_dataset[chat_id]["logs"].append(alf_log)

        # Sort turns by time_in and determine if they were sent to the user
        if not self.channel_id:
            for _, chat_data in raw_dataset.items():
                chat_data["logs"].sort(key=lambda x: x.time_in)
                for i, alf_log in enumerate(chat_data["logs"]):
                    alf_log.sent = all(alf_log.time_out <= next_alf_log.time_in for next_alf_log in chat_data["logs"][i+1:])

        # Convert to AlfLog and AlfTurn models
        chats = []
        for chat_id, chat_data in raw_dataset.items():
            channel_id = self.channel_id or "local"
            alf_chat = ALFChat(
                channel_id=channel_id,
                chat_id=chat_id,
                logs=chat_data["logs"]
            )
            chats.append(alf_chat)
            
        logs = []
        for alf_chat in chats:
            logs.extend(alf_chat.logs)
            
        return chats, logs


    def split_data_by_response_type(self):
        """
        Split the dataset into knowledge-based and function-call datasets.
        """
        turns_kb = []
        turns_fc = []
        
        for alf_log in self.logs:
            if alf_log.with_knowledge and alf_log.sent:
                turns_kb.append(alf_log)
            else:
                turns_fc.append(alf_log)
                        
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
        for alf_chat in self.chats:
            alf_logs = alf_chat.logs
            if alf_logs and alf_logs[-1].response_type == 'open_chat':
                for alf_log in alf_logs:
                    if alf_log.sent and alf_log.response_type in ['faq', 'rag']:
                        if datetime.strptime(alf_log.time_in, "%Y-%m-%d %H:%M:%S.%f%z") - datetime.strptime(alf_log.time_in, "%Y-%m-%d %H:%M:%S.%f%z") < timedelta(hours=24):
                            unsatisfied_chats.add(alf_chat.chat_id)
                            break
                            
        unsatisfied_logs = {k: v for k, v in self.dataset.items() if k in unsatisfied_chats}
        logger.info(f"Number of unsatisfied chats: {len(unsatisfied_logs)}")
        return unsatisfied_logs
    
    def detect_language(self):
        """
        Detect the language of the chat.
        """
        for alf_log in self.logs:
            alf_log.language = detect_language(alf_log.summary)
