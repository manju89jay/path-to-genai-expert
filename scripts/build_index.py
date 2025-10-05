import os, re, datetime, pathlib

LOG_DIR = pathlib.Path("logs")
readme_path = LOG_DIR / "README.md"
week_files = sorted([p for p in LOG_DIR.glob("week-*.md") if p.is_file()])

index_lines = []
for p in week_files:
    text = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^#\s*(.+)$", text, flags=re.M)
    title = m.group(1).strip() if m else p.stem
    index_lines.append(f"- [{title}]({p.as_posix()})")

with readme_path.open("r", encoding="utf-8") as f:
    content = f.read()

start_tag = "<!-- INDEX:START -->"
end_tag = "<!-- INDEX:END -->"
pre = content.split(start_tag)[0] + start_tag + "\n"
post = "\n" + end_tag + content.split(end_tag)[1]

new_content = pre + "\n".join(index_lines) + post

if new_content != content:
    readme_path.write_text(new_content, encoding="utf-8")
    print("Updated logs index.")
else:
    print("No changes to logs index.")
