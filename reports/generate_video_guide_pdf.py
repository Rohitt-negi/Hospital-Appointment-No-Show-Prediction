"""
Script to generate a professional 2-page PDF guide for Rohit Singh Negi's
2-Minute Video Presentation for Algions AI Labs.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def create_pdf(output_path='reports/2_Minute_Video_Walkthrough_Guide.pdf'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 0.5 inch margins so everything fits comfortably on 2 pages
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor('#1e3a8a')    # Dark Blue
    secondary_color = colors.HexColor('#2563eb')  # Royal Blue
    dark_neutral = colors.HexColor('#0f172a')     # Slate 900
    text_muted = colors.HexColor('#475569')       # Slate 600
    card_bg = colors.HexColor('#f8fafc')          # Light Slate
    accent_green = colors.HexColor('#047857')     # Emerald Green
    accent_orange = colors.HexColor('#b45309')    # Amber

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        alignment=0
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_muted
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=dark_neutral
    )

    bold_body = ParagraphStyle(
        'BoldBody',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=dark_neutral
    )

    script_spoken = ParagraphStyle(
        'ScriptSpoken',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1e293b')
    )

    script_action = ParagraphStyle(
        'ScriptAction',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=secondary_color
    )

    story = []

    # -------------------------------------------------------------
    # PAGE 1: HEADER & HIGH-LEVEL OVERVIEW
    # -------------------------------------------------------------
    header_data = [
        [
            Paragraph("<b>2-Minute Video Presentation Guide & Script</b>", title_style),
            Paragraph("<b>Candidate:</b> Rohit Singh Negi<br/><b>Reg No:</b> 12311143<br/><b>Assessment:</b> Algions AI Labs", ParagraphStyle('HeaderRight', parent=body_style, alignment=2, fontSize=8.5, leading=11))
        ]
    ]
    t_header = Table(header_data, colWidths=[380, 160])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Project Domain:</b> Hospital Outpatient Operations Decision Support &bull; <b>Target Duration:</b> Exactly 2 Minutes (120 Seconds)", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=2, spaceAfter=8))

    # SECTION 1: PROJECT OVERVIEW (WHAT TO KNOW BEFORE TALKING)
    story.append(Paragraph("1. High-Level Project Overview (What You Need to Know)", section_heading))
    
    overview_table_data = [
        [
            Paragraph("<b>Component</b>", bold_body),
            Paragraph("<b>High-Level Overview (What to Keep in Mind)</b>", bold_body)
        ],
        [
            Paragraph("<b>What is this project?</b>", bold_body),
            Paragraph("An AI decision-support tool for hospital receptionists and clinic operations. It predicts if a patient is likely to miss a scheduled outpatient appointment so staff can send targeted reminders or manage backup schedule buffers.", body_style)
        ],
        [
            Paragraph("<b>Where did the data come from?</b>", bold_body),
            Paragraph("A de-identified synthetic dataset of <b>15,000 appointment records</b> modeled on the standard public <b>Kaggle Medical Appointment No-Shows</b> benchmark. Strictly compliant with HIPAA: <b>zero sensitive clinical data</b> (no diagnoses, medical histories, or medications). Uses only booking-time variables (lead time, age, distance, SMS sent, and past attendance history).", body_style)
        ],
        [
            Paragraph("<b>What ML models were used?</b>", bold_body),
            Paragraph("Evaluated 4 algorithms: <b>Baseline Logistic Regression</b> (unweighted), <b>Balanced Logistic Regression</b> (class-weighted), <b>Random Forest</b>, and <b>Histogram Gradient Boosting</b>. Balanced Logistic Regression was chosen as the primary model for its clean probability calibration, zero latency, and explainability.", body_style)
        ],
        [
            Paragraph("<b>Key Breakthrough & Business Cost</b>", bold_body),
            Paragraph("Because an empty doctor slot costs <b>~$180</b> while a reminder call costs only <b>~$15</b>, missing a no-show is 12x worse! By adjusting our decision threshold from 0.50 to <b>0.35</b>, our model captures <b>97.7% of all no-shows</b>, cutting wasted clinic capacity costs by <b>76%</b> ($43.01 down to $10.18 per appointment).", body_style)
        ],
        [
            Paragraph("<b>Governance & Ethics</b>", bold_body),
            Paragraph("This is purely a <b>decision-support tool</b> for administrative reminders. It is <b>never used to deny care</b>, alter treatments, or triage clinical urgency. High risk triggers proactive support (like transport help or rescheduling).", body_style)
        ]
    ]

    t_overview = Table(overview_table_data, colWidths=[130, 410])
    t_overview.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, card_bg]),
    ]))
    story.append(t_overview)
    story.append(Spacer(1, 8))

    # SECTION 2: RECORDING SETUP
    story.append(Paragraph("2. Quick Setup Before You Press Record", section_heading))
    setup_data = [
        [
            Paragraph("<b>Step 1: Start Server</b>", bold_body),
            Paragraph("Open Terminal in project folder and run: <code>python3 app/app.py</code> (Server runs on <code>http://127.0.0.1:5001</code>).", body_style)
        ],
        [
            Paragraph("<b>Step 2: Prepare Browser</b>", bold_body),
            Paragraph("Open <code>http://127.0.0.1:5001</code> in Chrome or Safari. Ensure your name <b>Rohit Singh Negi</b> and <b>Reg No: 12311143</b> are clearly visible in the top navbar.", body_style)
        ],
        [
            Paragraph("<b>Step 3: Recording Tool</b>", bold_body),
            Paragraph("Use <b>QuickTime Player</b> (File &rarr; New Screen Recording), <b>Loom</b>, or <b>OBS</b>. Select full screen or browser window with clear microphone audio.", body_style)
        ]
    ]
    t_setup = Table(setup_data, colWidths=[130, 410])
    t_setup.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0fdf4')),
    ]))
    story.append(t_setup)

    # PAGE BREAK TO PAGE 2 (EXACT 2-MINUTE TIMED SCRIPT)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # PAGE 2: 2-MINUTE TIMED SCRIPT & CHEAT SHEET
    # -------------------------------------------------------------
    story.append(Paragraph("3. The Exact 2-Minute (120-Second) Spoken Script & Screen Actions", section_heading))
    story.append(Paragraph("<i>Speak at a natural, steady pace. You don't have to memorize every single word—just follow the bullet points!</i>", subtitle_style))
    story.append(Spacer(1, 4))

    script_table_data = [
        [
            Paragraph("<b>Time</b>", bold_body),
            Paragraph("<b>Screen Action (What to Show)</b>", bold_body),
            Paragraph("<b>Exact Words to Speak (What to Say)</b>", bold_body)
        ],
        [
            Paragraph("<b>0:00 - 0:25</b><br/>(25 sec)", bold_body),
            Paragraph("<b>Website Home Screen</b><br/>Show browser at <code>http://127.0.0.1:5001</code> with navbar displaying your name and Reg No.", script_action),
            Paragraph('"Hello everyone. My name is <b>Rohit Singh Negi</b>, registration number <b>12311143</b>. Today I am presenting my project: <b>Hospital Appointment No-Show Prediction</b>.<br/><br/>In outpatient hospital operations, when a patient misses an appointment without notice, doctors sit idle and valuable clinic time is wasted. Our goal is to build an operational decision-support tool that predicts no-show risks at the time of booking, helping clinic staff direct phone reminders and organize scheduling buffers."', script_spoken)
        ],
        [
            Paragraph("<b>0:25 - 0:50</b><br/>(25 sec)", bold_body),
            Paragraph("<b>Model Analytics Tab</b><br/>Click the <i>'Model Analytics & Validation'</i> tab. Hover over Lead Time and Age plots.", script_action),
            Paragraph('"For the dataset, respecting healthcare privacy and HIPAA standards, we used a de-identified outpatient dataset of 15,000 records modeled after the public Kaggle Medical Appointment benchmark. It strictly excludes sensitive clinical diagnoses.<br/><br/>We engineered booking-time features like booking lead time, patient age, distance from clinic, SMS reminder status, and prior visit history. We used a strict <b>chronological split</b> to ensure zero data leakage from future visits into training."', script_spoken)
        ],
        [
            Paragraph("<b>0:50 - 1:15</b><br/>(25 sec)", bold_body),
            Paragraph("<b>Cost Curve & Metrics</b><br/>Point to the <i>'Cost Optimization Curve'</i> and metric cards (ROC-AUC 0.698, Recall 97.7%).", script_action),
            Paragraph('"To handle the 4:1 class imbalance, we trained multiple models: baseline Logistic Regression, Balanced Random Forest, HistGradientBoosting, and Balanced Logistic Regression.<br/><br/>Because an empty doctor slot costs about $180 while a quick phone call costs only $15, missing a no-show is 12 times more expensive! By optimizing our decision threshold to <b>0.35</b>, our model catches <b>97.7% of all no-shows</b>, cutting wasted clinic costs by <b>76%</b>."', script_spoken)
        ],
        [
            Paragraph("<b>1:15 - 1:45</b><br/>(30 sec)", bold_body),
            Paragraph("<b>Live Demo: Single & Batch</b><br/>1. Switch to <i>'Single Patient'</i> tab.<br/>2. Click <b>'Load High Risk Sample'</b> &rarr; <b>'Evaluate'</b>.<br/>3. Switch to <i>'Batch Schedule'</i> tab &rarr; Click <b>'Run Sample Clinic Day'</b>.", script_action),
            Paragraph('"Now let\'s see the live web dashboard. If I load a high-risk sample—a 23-year-old booking 28 days ahead with past missed visits—the model outputs a <b>68% risk probability</b>, explaining the top risk drivers and recommending a direct telephone call 48 hours prior.<br/><br/>In the <b>Batch Scoring</b> tab, clinic staff can evaluate an entire day\'s appointment roster in one click to prioritize who needs reminders."', script_spoken)
        ],
        [
            Paragraph("<b>1:45 - 2:00</b><br/>(15 sec)", bold_body),
            Paragraph("<b>Governance Tab & Closing</b><br/>Click <i>'Governance & Limitations'</i> tab, then show VS Code / terminal briefly.", script_action),
            Paragraph('"Finally, this system is strictly a decision-support aid—it is never used to deny care or triage medical urgency. The project includes 12 automated unit tests passing at 100% and is published on GitHub. Thank you!"', script_spoken)
        ]
    ]

    t_script = Table(script_table_data, colWidths=[65, 155, 320])
    t_script.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, card_bg]),
    ]))
    story.append(t_script)
    story.append(Spacer(1, 6))

    # SECTION 4: CHEAT SHEET SUMMARY
    story.append(Paragraph("4. Quick 3-Point Mental Cheat Sheet (If You Get Stuck)", section_heading))
    cheat_data = [
        [
            Paragraph("<b>1. Problem:</b> Hospital no-shows waste physician slots ($180 loss). A reminder phone call is cheap ($15).", body_style),
            Paragraph("<b>2. Data & Model:</b> 15k de-identified records, no clinical data. Balanced Logistic Regression catches 97.7% of no-shows.", body_style),
            Paragraph("<b>3. App Demo:</b> Live Flask dashboard with single evaluation, risk gauge, local explainability, and batch roster scoring.", body_style),
        ]
    ]
    t_cheat = Table(cheat_data, colWidths=[180, 180, 180])
    t_cheat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eff6ff')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#bfdbfe')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_cheat)

    doc.build(story)
    print(f"PDF generated successfully at: {output_path}")


if __name__ == '__main__':
    create_pdf()
