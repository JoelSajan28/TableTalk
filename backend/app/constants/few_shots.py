FEW_SHOTS = """
Examples (use exact table/column names):

# ---------- ROWS intent ----------
Q: first 3 tasks from build ->
SELECT category, task FROM racibuild ORDER BY category, task LIMIT 3

Q: total number of categories in run ->
SELECT COUNT(DISTINCT category) AS total FROM racirun

# ---------- SCHOOL: Row → Column (return column NAMES) ----------
Q: For Student_001, which classes are marked Studied? ->
SELECT student, GROUP_CONCAT(class, ', ') AS studied_classes
FROM (
  SELECT student, 'class_a' AS class FROM school_records WHERE student='Student_001' AND LOWER(COALESCE(class_a,'')) LIKE '%studied%'
  UNION ALL
  SELECT student, 'class_b' FROM school_records WHERE student='Student_001' AND LOWER(COALESCE(class_b,'')) LIKE '%studied%'
  UNION ALL
  SELECT student, 'class_c' FROM school_records WHERE student='Student_001' AND LOWER(COALESCE(class_c,'')) LIKE '%studied%'
  UNION ALL
  SELECT student, 'class_d' FROM school_records WHERE student='Student_001' AND LOWER(COALESCE(class_d,'')) LIKE '%studied%'
)
GROUP BY student
LIMIT 50

# ---------- SCHOOL: Column → Row (filter one column, return rows) ----------
Q: Who studied Class C? ->
SELECT student
FROM school_records
WHERE LOWER(COALESCE(class_c,'')) LIKE '%studied%'
ORDER BY student
LIMIT 50

# ---------- COMMENTS keyword across BOTH sheets (wrap union before outer ORDER BY) ----------
Q: find 'authority' mentions in comments ->
SELECT source, task, comments FROM (
  SELECT 'racibuild' AS source, task, comments FROM racibuild
  WHERE LOWER(COALESCE(comments,''))   LIKE LOWER('%authority%')
     OR LOWER(COALESCE(comments_1,'')) LIKE LOWER('%authority%')
     OR LOWER(COALESCE(comments_2,'')) LIKE LOWER('%authority%')
     OR LOWER(COALESCE(comments_3,'')) LIKE LOWER('%authority%')
     OR LOWER(COALESCE(comments_4,'')) LIKE LOWER('%authority%')
  UNION ALL
  SELECT 'racirun'  AS source, task, comments FROM racirun
  WHERE LOWER(COALESCE(comments,''))   LIKE LOWER('%authority%')
     OR LOWER(COALESCE(comments_1,'')) LIKE LOWER('%authority%')
     OR LOWER(COALESCE(comments_2,'')) LIKE LOWER('%authority%')
     OR LOWER(COALESCE(comments_3,'')) LIKE LOWER('%authority%')
     OR LOWER(COALESCE(comments_4,'')) LIKE LOWER('%authority%')
)
ORDER BY source, task
LIMIT 50

# ---------- RACI: Task → Teams across BOTH sheets (NO phantom columns; per-table column sets) ----------
Q: who is responsible for pam platform management ->
SELECT source, category, task, responsible_columns FROM (
  -- Build: only reference columns that exist in racibuild
  SELECT
    'racibuild' AS source,
    t.category,
    t.task,
    TRIM(
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.ibm_cloud,' ',''),'/','')),'r')>0 THEN 'ibm_cloud, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.fss__project_build,' ',''),'/','')),'r')>0 THEN 'fss__project_build, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.expert_labs_ftm,' ',''),'/','')),'r')>0 THEN 'expert_labs_ftm, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.ipcplatformbuild,' ',''),'/','')),'r')>0 THEN 'ipcplatformbuild, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.appops,' ',''),'/','')),'r')>0 THEN 'appops, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.ams_run,' ',''),'/','')),'r')>0 THEN 'ams_run, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.concentrix,' ',''),'/','')),'r')>0 THEN 'concentrix, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.platform_run,' ',''),'/','')),'r')>0 THEN 'platform_run, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.sos,' ',''),'/','')),'r')>0 THEN 'sos, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.mss,' ',''),'/','')),'r')>0 THEN 'mss, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.gbs_biso,' ',''),'/','')),'r')>0 THEN 'gbs_biso, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.canadian_bank,' ',''),'/','')),'r')>0 THEN 'canadian_bank, ' END,'')
    , ', ') AS responsible_columns
  FROM racibuild AS t
  WHERE LOWER(t.task)     LIKE LOWER('%pam platform management%')
     OR LOWER(t.category) LIKE LOWER('%pam platform management%')

  UNION ALL

  -- Run: only reference columns that exist in racirun (includes 'iris')
  SELECT
    'racirun' AS source,
    t.category,
    t.task,
    TRIM(
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.ibm_cloud,' ',''),'/','')),'r')>0 THEN 'ibm_cloud, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.fss__project_build,' ',''),'/','')),'r')>0 THEN 'fss__project_build, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.expert_labs_ftm,' ',''),'/','')),'r')>0 THEN 'expert_labs_ftm, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.appops,' ',''),'/','')),'r')>0 THEN 'appops, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.ams_run,' ',''),'/','')),'r')>0 THEN 'ams_run, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.concentrix,' ',''),'/','')),'r')>0 THEN 'concentrix, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.platform_run,' ',''),'/','')),'r')>0 THEN 'platform_run, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.sos,' ',''),'/','')),'r')>0 THEN 'sos, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.mss,' ',''),'/','')),'r')>0 THEN 'mss, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.gbs_biso,' ',''),'/','')),'r')>0 THEN 'gbs_biso, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.iris,' ',''),'/','')),'r')>0 THEN 'iris, ' END,'') ||
      COALESCE(CASE WHEN INSTR(LOWER(REPLACE(REPLACE(t.canadian_bank,' ',''),'/','')),'r')>0 THEN 'canadian_bank, ' END,'')
    , ', ') AS responsible_columns
  FROM racirun AS t
  WHERE LOWER(t.task)     LIKE LOWER('%pam platform management%')
     OR LOWER(t.category) LIKE LOWER('%pam platform management%')
)
WHERE COALESCE(responsible_columns,'') <> ''
ORDER BY source, task
LIMIT 50

# ---------- RACI: Team → Tasks across BOTH sheets ----------
Q: what are the tasks expert labs responsible for ->
SELECT source, category, task FROM (
  SELECT 'racibuild' AS source, category, task
  FROM racibuild
  WHERE INSTR(LOWER(REPLACE(REPLACE(expert_labs_ftm,' ',''),'/','')),'r')>0
  UNION ALL
  SELECT 'racirun' AS source, category, task
  FROM racirun
  WHERE INSTR(LOWER(REPLACE(REPLACE(expert_labs_ftm,' ',''),'/','')),'r')>0
)
ORDER BY source, category, task
LIMIT 50

IMPORTANT:
 - When adding an outer WHERE or ORDER BY to combined results, WRAP the UNION ALL in a subquery.
 - Only use UNION ALL for identical projections across compatible tables.
 - Do NOT cartesian join unrelated tables. Use explicit JOIN ... ON only with real keys.
 - Never ORDER BY a column not present in the SELECT list.
 - Never reference columns that do not exist in a given table; build per-table expressions accordingly.
"""
