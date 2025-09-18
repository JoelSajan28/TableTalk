import re 

_FORBIDDEN = re.compile(
    r"\b(DELETE|UPDATE|INSERT|REPLACE|ALTER|DROP|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM|CREATE)\b",
    re.I,
)
_START_OK = re.compile(r"^\s*(SELECT|WITH|COUNT|COALESCE|INSTR|CASE|WHEN|LOWER|REPLACE|THEN|TRIM|END|FROM|WHERE|LIMIT)\b", re.I)
_SQL_FENCE = re.compile(r"```sql\s*(.*?)```", re.I | re.S)
_THINK_TAGS = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.I | re.S)
_NUM_RE = re.compile(r'\b(?:first|top)\s+(\d+)\b', re.I)
_SMALLTALK = re.compile(r'^(hi|hello|hey|thanks|thank you|bye)\b', re.I)
_FIRST_NO_NUM = re.compile(r'\bfirst\b(?!\s*\d)', re.I)
_NUM_PHRASE  = re.compile(r'\b(?:first|top)\s+(\d+)\b', re.I)
