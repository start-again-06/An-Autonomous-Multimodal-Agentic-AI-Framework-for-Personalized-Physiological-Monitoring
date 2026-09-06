from pathlib import Path
import platform,sys
p=Path('outputs/reproduction_report.md'); p.write_text(f'''# AMPA-HR Reproduction Report\n\nThis artifact reports values calculated by the included implementation; paper-reported values are not hard-coded.\n\n- Python: {sys.version.split()[0]}\n- Platform: {platform.platform()}\n- Dataset: synthetic physiological demo\n- Seed: 42\n- Note: PPG-DaLiA reproduction requires the user to obtain the dataset under its applicable terms.\n'''); print(p)
