"""Human-readable, offline summaries with escaped untrusted labels and text."""

from html import escape
import json


def _text(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, allow_nan=False)
    return str(value)


def _sections(report):
    results, decision = report["results"], report["decision"]
    sections = []
    if "data" in results:
        data = results["data"]
        sections.append(("Data quality", ["Status", "Findings"], list(data["counts"].items())))
    if "evaluation" in results:
        rows = []
        for name, model in results["evaluation"]["models"].items():
            rows.append([name, model["support"]["rows"], model["classification"]["threshold"],
                         model["classification"]["f1"], model["ranking"]["roc_auc"],
                         model["probability_scores"]["brier_score"], model["calibration"]["expected_calibration_error"]])
        sections.append(("Model evaluation", ["Model", "Rows", "Threshold", "F1", "ROC AUC", "Brier loss", "Calibration gap"], rows))
    if "segments" in results:
        rows = []
        for name, model in results["segments"]["models"].items():
            for column in model["segments"]:
                for group in column["groups"]:
                    rows.append([name, column["column"], "(missing)" if group["is_missing"] else group["value"],
                                 group["support"]["rows"], group["metrics"]["f1"],
                                 group["difference_from_overall"]["f1"], len(group["warnings"])])
        sections.append(("Segment performance", ["Model", "Column", "Group", "Rows", "F1", "Supported F1 difference", "Warnings"], rows))
    if "drift" in results:
        drift = results["drift"]["comparison"]
        rows = [[name, feature["kind"], feature.get("ks_distance", feature.get("total_variation")),
                 feature["missingness"]["fraction_difference"], feature.get("new_category_fraction")]
                for name, feature in drift["features"].items()]
        sections.append(("Feature distributions", ["Feature", "Kind", "Distance", "Missing fraction difference", "New category fraction"], rows))
        predictions = drift["predictions"]
        if predictions["status"] == "measured":
            rows = [["Empirical CDF distance", predictions["ks_distance"]],
                    ["Histogram total variation", predictions["histogram"]["total_variation"]],
                    ["Mean probability difference", predictions["mean_probability_difference"]],
                    ["Predicted positive fraction difference", predictions["predicted_positive_fraction_difference"]]]
        else:
            rows = [["Skipped", predictions["reason"]]]
        sections.append(("Prediction distributions", ["Measurement", "Value"], rows))
        performance = drift["performance"]
        if performance["status"] == "measured":
            rows = [[name, value, performance["metric_direction"][name],
                     performance["difference_undefined_reasons"].get(name, "Supported")]
                    for name, value in performance["difference"].items()]
        else:
            rows = [["Skipped", None, "", performance["reason"]]]
        sections.append(("Labeled performance change", ["Metric", "Current minus reference", "Direction", "Support"], rows))
    sections.append(("Configured gates", ["Name", "Result path", "Value", "Minimum", "Maximum", "Status", "Reason"],
                     [[g["name"], g["path"], g["value"], g.get("minimum", "Unbounded"),
                       g.get("maximum", "Unbounded"), g["status"], g["reason"] or ""] for g in decision["gates"]]))
    sections.append(("Review items", ["Location", "Detail"], [[item["path"], item["detail"]] for item in decision["review_items"]]))
    sections.append(("Data failures", ["Location", "Detail"], [[item["path"], item["detail"]] for item in decision["data_failures"]]))
    return sections


def render_reports(report):
    """Return Markdown and self-contained HTML; no remote resources or scripts."""
    decision = report["decision"]
    title = "ModelAudit report"
    intro = [f"Outcome: {decision['status'].upper()}",
             f"Created: {report['context']['created_at']}",
             f"Enabled checks: {', '.join(report['results'])}",
             f"Review items: {len(decision['review_items'])}; data failures: {len(decision['data_failures'])}; gates: {len(decision['gates'])}.",
             decision["meaning"],
             "Unavailable means missing or unsupported evidence, never zero. Differences are current minus reference unless a segment comparison explicitly says otherwise.",
             "Feature/prediction drift is descriptive; no production chronology or causality is inferred. Full evidence and support rules are in report.json and the individual check files."]
    markdown = [f"# {title}", ""]
    html = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">',
            f'<title>{title}</title><style>body{{font:16px/1.5 system-ui,sans-serif;color:#172536;background:#f4f7fa;margin:0;padding:32px}}',
            'main{max-width:1200px;margin:auto}section,header{background:white;padding:24px;border:1px solid #d6e0ea;border-radius:12px;margin-bottom:20px}',
            'h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #d6e0ea;overflow-wrap:anywhere}',
            'th{background:#eaf1f7}.table{overflow-x:auto}p{max-width:95ch}</style></head><body><main><header>',
            f'<h1>{title}</h1>']
    for line in intro:
        markdown.extend([escape(line), ""])
        html.append(f"<p>{escape(line)}</p>")
    html.append("</header>")
    for heading, columns, rows in _sections(report):
        markdown.extend([f"## {heading}", ""])
        html.append(f"<section><h2>{escape(heading)}</h2>")
        if not rows:
            markdown.extend(["None.", ""])
            html.append("<p>None.</p></section>")
            continue
        def cell(value):
            text = escape(_text(value)).replace("\\", "\\\\")
            for character in "[]*_":
                text = text.replace(character, "\\" + character)
            return text.replace("|", "&#124;").replace("`", "&#96;").replace("\r", " ").replace("\n", " ")
        markdown.append("| " + " | ".join(columns) + " |")
        markdown.append("| " + " | ".join("---" for _ in columns) + " |")
        html.append('<div class="table"><table><thead><tr>' + ''.join(f"<th>{escape(c)}</th>" for c in columns) + '</tr></thead><tbody>')
        for row in rows:
            markdown.append("| " + " | ".join(cell(value) for value in row) + " |")
            html.append("<tr>" + "".join(f"<td>{escape(_text(value))}</td>" for value in row) + "</tr>")
        markdown.append("")
        html.append("</tbody></table></div></section>")
    html.append("</main></body></html>\n")
    return "\n".join(markdown), "\n".join(html)
