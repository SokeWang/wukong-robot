# -*- coding: utf-8 -*-
from .sdk import unit
from robot import logging
from abc import ABCMeta, abstractmethod
from agno.models.openai import OpenAIChat
from agno.agent import Agent
from agno.storage.agent.sqlite import SqliteAgentStorage
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum, auto

logger = logging.getLogger(__name__)


class AbstractNLU(object):
    """
    Generic parent class for all NLU engines
    """

    __metaclass__ = ABCMeta

    @classmethod
    def get_config(cls):
        return {}

    @classmethod
    def get_instance(cls):
        profile = cls.get_config()
        instance = cls(**profile)
        return instance

    @abstractmethod
    def parse(self, query, **args):
        """
        进行 NLU 解析

        :param query: 用户的指令字符串
        :param **args: 可选的参数
        """
        return None

    @abstractmethod
    def getIntent(self, parsed):
        """
        提取意图

        :param parsed: 解析结果
        :returns: 意图数组
        """
        return None

    @abstractmethod
    def hasIntent(self, parsed, intent):
        """
        判断是否包含某个意图

        :param parsed: 解析结果
        :param intent: 意图的名称
        :returns: True: 包含; False: 不包含
        """
        return False

    @abstractmethod
    def getSlots(self, parsed, intent):
        """
        提取某个意图的所有词槽

        :param parsed: 解析结果
        :param intent: 意图的名称
        :returns: 词槽列表。你可以通过 name 属性筛选词槽，
        再通过 normalized_word 属性取出相应的值
        """
        return None

    @abstractmethod
    def getSlotWords(self, parsed, intent, name):
        """
        找出命中某个词槽的内容

        :param parsed: 解析结果
        :param intent: 意图的名称
        :param name: 词槽名
        :returns: 命中该词槽的值的列表。
        """
        return None

    @abstractmethod
    def getSay(self, parsed, intent):
        """
        提取回复文本

        :param parsed: 解析结果
        :param intent: 意图的名称
        :returns: 回复文本
        """
        return ""


class UnitNLU(AbstractNLU):
    """
    百度UNIT的NLU API.
    """

    SLUG = "unit"

    def __init__(self):
        super(self.__class__, self).__init__()

    @classmethod
    def get_config(cls):
        """
        百度UNIT的配置

        无需配置，所以返回 {}
        """
        return {}

    def parse(self, query, **args):
        """
        使用百度 UNIT 进行 NLU 解析

        :param query: 用户的指令字符串
        :param **args: UNIT 的相关参数
            - service_id: UNIT 的 service_id
            - api_key: UNIT apk_key
            - secret_key: UNIT secret_key
        :returns: UNIT 解析结果。如果解析失败，返回 None
        """
        if (
            "service_id" not in args
            or "api_key" not in args
            or "secret_key" not in args
        ):
            logger.critical(f"{self.SLUG} NLU 失败：参数错误！", stack_info=True)
            return None
        return unit.getUnit(
            query, args["service_id"], args["api_key"], args["secret_key"]
        )

    def getIntent(self, parsed):
        """
        提取意图

        :param parsed: 解析结果
        :returns: 意图数组
        """
        return unit.getIntent(parsed)

    def hasIntent(self, parsed, intent):
        """
        判断是否包含某个意图

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :returns: True: 包含; False: 不包含
        """
        return unit.hasIntent(parsed, intent)

    def getSlots(self, parsed, intent):
        """
        提取某个意图的所有词槽

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :returns: 词槽列表。你可以通过 name 属性筛选词槽，
        再通过 normalized_word 属性取出相应的值
        """
        return unit.getSlots(parsed, intent)

    def getSlotWords(self, parsed, intent, name):
        """
        找出命中某个词槽的内容

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :param name: 词槽名
        :returns: 命中该词槽的值的列表。
        """
        return unit.getSlotWords(parsed, intent, name)

    def getSlotOriginalWords(self, parsed, intent, name):
        """
        找出命中某个词槽的原始内容

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :param name: 词槽名
        :returns: 命中该词槽的值的列表。
        """
        return unit.getSlotOriginalWords(parsed, intent, name)

    def getSay(self, parsed, intent):
        """
        提取 UNIT 的回复文本

        :param parsed: UNIT 解析结果
        :param intent: 意图的名称
        :returns: UNIT 的回复文本
        """
        return unit.getSay(parsed, intent)

class Intent(str, Enum):
    """意图枚举"""
    BUILT_POEM = "BUILT_POEM"  # 写诗
    WEATHER = "WEATHER"  # 天气
    TIME = "TIME"  # 时间
    MUSIC = "MUSIC"  # 音乐
    MUSICRANK = "MUSICRANK"  # 音乐排行
    CHANGE_TO_NEXT = "CHANGE_TO_NEXT"  # 下一首
    CHANGE_TO_LAST = "CHANGE_TO_LAST"  # 上一首
    CHANGE_VOL = "CHANGE_VOL"  # 调整音量
    CONTINUE = "CONTINUE"  # 继续播放
    CLOSE_MUSIC = "CLOSE_MUSIC"  # 关闭音乐
    PAUSE = "PAUSE"  # 暂停
    CHECK_REMIND = "CHECK_REMIND"  # 查询提醒
    DELETE_REMIND = "DELETE_REMIND"  # 删除提醒
    SET_REMIND = "SET_REMIND"  # 设置提醒
    HASS_INDEX = "HASS_INDEX"  # 选择序号
    # 添加更多意图...

class SlotName(str, Enum):
    """词槽名称枚举"""
    # 通用词槽
    CITY = "city"  # 城市
    DATE = "date"  # 日期
    TIME = "time"  # 时间
    PERIOD = "period"  # 时段
    
    # 音乐相关词槽
    SONG = "song"  # 歌曲
    ARTIST = "artist"  # 艺术家
    
    # 诗歌相关词槽
    POEM_TYPE = "poem_type"  # 诗歌类型
    POEM_THEME = "poem_theme"  # 诗歌主题
    POEM_STYLE = "poem_style"  # 诗歌风格
    
    # 音量控制词槽
    USER_D = "user_d"  # 音量调整方向 (--HIGHER-- 或其他)
    USER_VD = "user_vd"  # 音量调整方向 (--LOUDER-- 或其他)
    
    # 提醒相关词槽
    USER_REMIND_TIME = "user_remind_time"  # 提醒时间
    USER_WILD_CONTENT = "user_wild_content"  # 提醒内容
    
    # 序号相关词槽
    USER_INDEX = "user_index"  # 用户选择的序号

class SlotValue(BaseModel):
    """词槽值"""
    name: SlotName = Field(..., description="词槽名称")
    original_word: str = Field(..., description="原始词槽值")
    normalized_word: str = Field(..., description="标准化后的词槽值")
    value: Optional[str] = Field(None, description="处理后的值，如日期、数字等的字符串表示")

class IntentMatchResult(BaseModel):
    """意图匹配结果"""
    intent: Intent = Field(..., description="匹配到的意图")
    confidence: float = Field(..., description="匹配置信度")
    slots: List[SlotValue] = Field(default_factory=list, description="提取的词槽列表")
    say: Optional[str] = Field(None, description="回复文本")
    reasoning: str = Field(..., description="匹配原因")


class AgnoNLU(AbstractNLU):
    """
    基于Agno的NLU实现
    """
    
    SLUG = "agno"
    
    def __init__(self):
        """
        初始化AgnoNLU
        
        :param model_name: 使用的模型名称
        :param intents: 支持的意图列表
        """
        super(self.__class__, self).__init__()
        self.gpt_4o_mini = OpenAIChat(
            id="gpt-4o-mini",
            api_key="sk-KbdOsQkrlY8GWSPi78Ij8CTYc6KE39lfFJ5ZpO1HapmF8olL",
            base_url="https://www.DMXapi.com/v1"
        )
        self.deepseek_v3 = OpenAIChat(
            id="deepseek_v3",
            api_key="bce-v3/ALTAK-f7eRt1AmOEQDKbmVjqcTh/b4a62ce5423caed709e9ac2686b7aa94a050554b",
            base_url="https://qianfan.baidubce.com/v2"
        )
        
        # 意图到词槽的映射
        INTENT_SLOTS = {
            Intent.BUILT_POEM.value: [SlotName.POEM_TYPE.value, SlotName.POEM_THEME.value, SlotName.POEM_STYLE.value],
            Intent.WEATHER.value: [SlotName.CITY.value, SlotName.DATE.value, SlotName.PERIOD.value],
            Intent.MUSIC.value: [SlotName.SONG.value, SlotName.ARTIST.value],
            Intent.CHANGE_VOL.value: [SlotName.USER_D.value, SlotName.USER_VD.value],
            Intent.SET_REMIND.value: [SlotName.USER_REMIND_TIME.value, SlotName.USER_WILD_CONTENT.value],
            Intent.HASS_INDEX.value: [SlotName.USER_INDEX.value],
            # 其他意图到词槽的映射...
        }

        # 创建NLU代理
        self.nlu_agent = Agent(
            model=self.gpt_4o_mini,
            description="你是一个专门进行用户意图识别的助手。",
            instructions=[
                "你是一个专门进行用户意图识别的助手。",
                "你需要根据用户的输入，判断用户的意图，并提取相关的词槽。",
                "支持的意图包括：" + ", ".join([intent.value for intent in Intent]),
                "每个意图有特定的词槽：",
                *[f"- {intent}: {', '.join(slots)}" for intent, slots in INTENT_SLOTS.items()],
                "请确保提取的词槽名称与对应意图的有效词槽匹配。"
            ],
            storage=SqliteAgentStorage(table_name="nlu_agent", db_file="temp/agents.db"),
            response_model=IntentMatchResult,
            structured_outputs=True,
            add_history_to_messages=True,
            add_datetime_to_instructions=True,
            monitoring=True
        )
    
    @classmethod
    def get_config(cls):
        """
        获取配置
        """
        return {}
    
    def parse(self, query, **args):
        """
        使用Agno进行NLU解析
        
        :param query: 用户的指令字符串
        :param **args: 可选参数
        :returns: 解析结果
        """
        try:

            # 运行NLU代理
            response = self.nlu_agent.run(query)
            
            if not response or not response.content:
                logger.warning("NLU解析返回空结果")
                return None
            
            return response.content
            
        except Exception as e:
            logger.error(f"NLU解析失败: {str(e)}")
            return None
    
    def getIntent(self, parsed):
        """
        提取意图
        
        :param parsed: 解析结果
        :returns: 意图数组
        """
        if not parsed or not isinstance(parsed, IntentMatchResult):
            return ""
        
        return parsed.intent
    
    def hasIntent(self, parsed, intent):
        """
        判断是否包含某个意图
        
        :param parsed: 解析结果
        :param intent: 意图的名称
        :returns: True: 包含; False: 不包含
        """
        if not parsed or not isinstance(parsed, IntentMatchResult):
            return False
        
        # 支持字符串和枚举类型
        intent_value = intent.value if isinstance(intent, Intent) else intent
        return parsed.intent == intent_value
    
    def getSlots(self, parsed, intent):
        """
        提取某个意图的所有词槽
        
        :param parsed: 解析结果
        :param intent: 意图的名称
        :returns: 词槽列表
        """
        if not parsed or not isinstance(parsed, IntentMatchResult) or parsed.intent != intent:
            return []
        
        return list(parsed.slots)
    
    def getSlotWords(self, parsed, intent, name):
        """
        找出命中某个词槽的内容
        
        :param parsed: 解析结果
        :param intent: 意图的名称
        :param name: 词槽名
        :returns: 命中该词槽的标准化值列表
        """
        if not parsed or not isinstance(parsed, IntentMatchResult) or not self.hasIntent(parsed, intent):
            return []
        
        # 支持字符串和枚举类型
        slot_name = name.value if hasattr(name, 'value') else name
        return [slot.normalized_word for slot in parsed.slots if slot.name == slot_name]
    
    def getSlotOriginalWords(self, parsed, intent, name):
        """
        找出命中某个词槽的原始内容
        
        :param parsed: 解析结果
        :param intent: 意图的名称
        :param name: 词槽名
        :returns: 命中该词槽的原始值列表
        """
        if not parsed or not isinstance(parsed, IntentMatchResult) or not self.hasIntent(parsed, intent):
            return []
        
        # 支持字符串和枚举类型
        slot_name = name.value if hasattr(name, 'value') else name
        return [slot.original_word for slot in parsed.slots if slot.name == slot_name]
    
    def getSay(self, parsed, intent):
        """
        提取回复文本
        
        :param parsed: 解析结果
        :param intent: 意图的名称
        :returns: 回复文本
        """
        if not parsed or not isinstance(parsed, IntentMatchResult) or parsed.intent != intent:
            return ""
        
        return parsed.say or ""
    

def get_engine_by_slug(slug=None):
    """
    Returns:
        An NLU Engine implementation available on the current platform

    Raises:
        ValueError if no speaker implementation is supported on this platform
    """

    if not slug or type(slug) is not str:
        raise TypeError("无效的 NLU slug '%s'", slug)

    selected_engines = list(
        filter(
            lambda engine: hasattr(engine, "SLUG") and engine.SLUG == slug,
            get_engines(),
        )
    )

    if len(selected_engines) == 0:
        raise ValueError(f"错误：找不到名为 {slug} 的 NLU 引擎")
    else:
        if len(selected_engines) > 1:
            logger.warning(f"注意: 有多个 NLU 名称与指定的引擎名 {slug} 匹配")
        engine = selected_engines[0]
        logger.info(f"使用 {engine.SLUG} NLU 引擎")
        return engine.get_instance()


def get_engines():
    def get_subclasses(cls):
        subclasses = set()
        for subclass in cls.__subclasses__():
            subclasses.add(subclass)
            subclasses.update(get_subclasses(subclass))
        return subclasses

    return [
        engine
        for engine in list(get_subclasses(AbstractNLU))
        if hasattr(engine, "SLUG") and engine.SLUG
    ]
