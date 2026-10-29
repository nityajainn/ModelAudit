"""Offline experiment summaries; detector outcome and experiment success differ."""

from html import escape
import json


def render_experiments(summary):
    matched = sum(case["expectations_met"] for case in summary["scenarios"])
    title = "ModelAudit controlled experiments"
    introduction = [f"Scenarios matching expectations: {matched}/{len(summary['scenarios'])}.",
                    f"Recorded: {summary['context']['created_at']}",
                    "A matched expectation means the controlled experiment behaved as specified. It does not mean the corrupted data or model passed its audit.",
                    "These are deliberate offline transformations, not production incidents or newly trained models. Demonstration gate bounds are not recommended release thresholds.",
                    "Source data and saved models are preserved. Full aggregate measurements are in each case JSON file; the completion manifest hashes every output."]
    markdown = [f"# {title}", ""]
    html = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">',
            f'<title>{title}</title><style>body{{font:16px/1.5 system-ui,sans-serif;background:#f4f7fa;color:#172536;margin:0;padding:28px}}',
            'main{max-width:1100px;margin:auto}section,header{padding:24px;margin-bottom:20px;background:white;border:1px solid #d6e0ea;border-radius:12px}',
            'table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #d6e0ea;vertical-align:top}th{background:#eaf1f7}',
            '.table{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}summary{cursor:pointer}h1,h2{line-height:1.2}</style></head><body><main><header>',
            f'<h1>{title}</h1>']
    for paragraph in introduction:
        markdown.extend([escape(paragraph), ""])
        html.append(f"<p>{escape(paragraph)}</p>")
    html.append('</header><section><h2>Scenario outcomes</h2><div class="table"><table><thead><tr><th>Scenario</th><th>Audit outcome</th><th>Expectation matched</th></tr></thead><tbody>')
    markdown.extend(["| Scenario | Audit outcome | Expectation matched |", "| --- | --- | --- |"])

    def md(value):
        text = escape(str(value)).replace("\\", "\\\\")
        for character in "[]*_":
            text = text.replace(character, "\\" + character)
        return text.replace("|", "&#124;").replace("`", "&#96;").replace("\n", " ").replace("\r", " ")

    for case in summary["scenarios"]:
        values = [case["id"], case["audit_outcome"], "Yes" if case["expectations_met"] else "No"]
        markdown.append("| " + " | ".join(md(value) for value in values) + " |")
        html.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>")
    html.append("</tbody></table></div></section>")
    markdown.append("")
    for case in summary["scenarios"]:
        markdown.extend([f"## {md(case['id'])}", "", md(case["mutation"]), "", f"Evidence file: {md(case['file'])}", ""])
        html.append(f"<section><h2>{escape(case['id'])}</h2><p>{escape(case['mutation'])}</p><p>Evidence file: {escape(case['file'])}</p><ul>")
        for expectation in case["expectations"]:
            line = ("Matched: " if expectation["met"] else "NOT MATCHED: ") + expectation["description"]
            markdown.append("- " + md(line))
            html.append(f"<li>{escape(line)}</li>")
        details = json.dumps(case["evidence"], indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True)
        html.append(f"</ul><details><summary>Aggregate evidence</summary><pre>{escape(details)}</pre></details></section>")
        markdown.append("")
    html.append("</main></body></html>\n")
    return "\n".join(markdown), "\n".join(html)
