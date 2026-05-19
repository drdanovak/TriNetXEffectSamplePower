with tab_report:
    st.write("This report is intended to give users language they can adapt for methods checks, internal review, or manuscript planning.")

    report_lines = []
    line_break = chr(10)

    for _, r in summary.iterrows():
        finding_name = str(r.get("Finding", "Outcome"))
        narrative_text = str(r.get("Narrative Interpretation", ""))

        st.markdown("### " + finding_name)
        st.write(narrative_text)

        report_block = "## " + finding_name + line_break + line_break + narrative_text + line_break
        report_lines.append(report_block)

    report_text = line_break.join(report_lines)
    st.download_button(
        "Download narrative report as Markdown",
        data=report_text.encode("utf-8"),
        file_name="trinetx_interpretive_report.md",
        mime="text/markdown",
    )
