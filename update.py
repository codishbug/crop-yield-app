import streamlit as st
import pickle
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# -----------------------------
# Load dataset for dropdown options
# -----------------------------
try:
    df = pd.read_csv("crop_yield.csv")  
except:
    df = pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"],
        "Season": ["Kharif", "Rabi", "Summer", "Whole Year", "Kharif"],
        "State": ["Maharashtra", "Punjab", "Karnataka", "Uttar Pradesh", "Bihar"]
    })

crop_options = df["Crop"].unique().tolist()
season_options = df["Season"].unique().tolist()
state_options = df["State"].unique().tolist()

# -----------------------------
# Load trained pipeline
# -----------------------------
with open("pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_crop_yield(pipeline, crop, crop_year, season, state, area, production, rainfall, fertilizer, pesticide):
    input_data = pd.DataFrame([{
        "Crop": crop,
        "Crop_Year": crop_year,
        "Season": season,
        "State": state,
        "Area": area,
        "Production": production,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }])
    prediction = pipeline.predict(input_data)
    return prediction[0]

# -----------------------------
# Generate Dynamic Future Outlook (Updated with predicted_yield)
# -----------------------------
def generate_future_outlook(rainfall, fertilizer, pesticide, season, predicted_yield):
    insights = []

    # Rainfall analysis
    if rainfall < 800:
        insights.append("Rainfall is below the typical level. Irrigation support will be important.")
    elif rainfall > 1200:
        insights.append("Rainfall is higher than usual. Ensure proper drainage to avoid crop damage.")
    else:
        insights.append("Rainfall is within the optimal range for healthy crop growth.")

    # Fertilizer analysis
    if fertilizer < 100:
        insights.append("Fertilizer use is low. Balanced nutrition could boost yield.")
    elif fertilizer > 300:
        insights.append("Fertilizer usage is high. Soil testing is recommended to avoid wastage.")
    else:
        insights.append("Fertilizer usage is within recommended limits.")

    # Pesticide analysis
    if pesticide > 100:
        insights.append("Pesticide usage is high. Monitor resistance and environmental impact.")
    else:
        insights.append("Pesticide usage is under control.")

    # Yield-based remark
    if predicted_yield < 1.5:
        insights.append(f"Predicted yield is low for the {season} season. Improved irrigation, fertilizer, and pest management could increase yield.")
    elif predicted_yield < 2.5:
        insights.append(f"Predicted yield is moderate for the {season} season. Optimizing inputs could further improve yield.")
    else:
        insights.append(f"Predicted yield looks promising for the {season} season.")

    return insights

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("🌾 Indian Crop Yield Prediction App")
    st.write("Predict crop yield based on agricultural and environmental factors.")

    # Inputs without preselected values
    crop = st.selectbox("Crop", ["Select Crop"] + crop_options)
    season = st.selectbox("Season", ["Select Season"] + season_options)
    state = st.selectbox("State", ["Select State"] + state_options)

    crop_year = st.number_input("Crop Year", min_value=2000, max_value=2100, value=2025, step=1)
    area = st.number_input("Area (in hectares)", min_value=0.0, value=1000.0, step=100.0)
    production = st.number_input("Production (in tonnes)", min_value=0.0, value=500.0, step=10.0)
    rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1200.0, step=10.0)
    fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, value=200.0, step=5.0)
    pesticide = st.number_input("Pesticide (kg/ha)", min_value=0.0, value=50.0, step=1.0)

    # Ensure user selects crop, season, and state
    if st.button("📄 Generate Report"):
        if crop == "Select Crop" or season == "Select Season" or state == "Select State":
            st.error("Please select all fields before generating the report.")
            return

        predicted_yield = predict_crop_yield(
            pipeline, crop, crop_year, season, state,
            area, production, rainfall, fertilizer, pesticide
        )

        st.success(f"🌾 Predicted Yield: {predicted_yield:.2f} tonnes/hectare")

        # -----------------------------
        # Visualization Charts
        # -----------------------------
        st.subheader("📈 Visualization of Key Agricultural Indicators")
        fig, ax = plt.subplots()
        features = ["Area(ha)", "Production", "Rainfall", "Fertilizer", "Pesticide"]
        values = [area, production, rainfall, fertilizer, pesticide]
        ax.bar(features, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'])
        ax.set_ylabel("Value")
        ax.set_title("Crop Input Indicators")
        st.pyplot(fig)

        # -----------------------------
        # Recommendations
        # -----------------------------
        st.subheader("📝 Recommendations")
        st.markdown("""
        - Ensure optimal irrigation based on rainfall conditions.  
        - Use balanced fertilizer within recommended limits (100–300 kg/ha).  
        - Avoid overuse of pesticides; explore organic alternatives.  
        - Monitor soil health regularly and practice crop rotation.  
        - Use high-yield, climate-resilient seed varieties.  
        - Stay updated with government schemes and agricultural advisories.  
        """)

        # -----------------------------
        # Future Outlook (Dynamic)
        # -----------------------------
        st.subheader("🔮 Future Outlook")
        outlook = generate_future_outlook(rainfall, fertilizer, pesticide, season, predicted_yield)
        for o in outlook:
            st.markdown(f"- {o}")

        # -----------------------------
        # Useful Resources
        # -----------------------------
        st.subheader("📌 Useful Resources")
        st.markdown("""
        - [Indian Council of Agricultural Research (ICAR)](https://icar.org.in/)  
        - [Ministry of Agriculture & Farmers Welfare](https://agricoop.nic.in/)  
        - [FAO Crop Production Guidelines](https://www.fao.org/crop-production/en/)  
        """)

        # -----------------------------
        # PDF Report Generation
        # -----------------------------
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # Custom styles
        title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=24, textColor=colors.HexColor('#1A5276'), alignment=1)
        rec_heading_style = ParagraphStyle('RecHeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#5B2C6F'))
        outlook_heading_style = ParagraphStyle('OutlookHeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#E75480'))
        resource_heading_style = ParagraphStyle('ResourceHeadingStyle', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#1F618D'))
        normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=12, textColor=colors.black)

        # Title
        elements.append(Paragraph("Crop Yield Prediction Report", title_style))
        elements.append(Spacer(1, 12))
        elements.append(Spacer(1, 12))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"Date: {datetime.date.today()}", normal_style))
        elements.append(Spacer(1, 12))

        # Prediction
        prediction_color = colors.green if predicted_yield >= 1 else colors.red
        prediction_style = ParagraphStyle('PredictionStyle', parent=styles['Heading2'], fontSize=16, textColor=prediction_color, alignment=1)
        elements.append(Paragraph(f"Predicted Yield: {predicted_yield:.2f} tonnes/ha", prediction_style))
        elements.append(Spacer(1, 12))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BDC3C7")))
        elements.append(Spacer(1, 12))

        # Input Details Table
        data = [
            ["Attribute", "Value"],
            ["Crop", crop],
            ["Year", crop_year],
            ["Season", season],
            ["State", state],
            ["Area (ha)", area],
            ["Production (t)", production],
            ["Rainfall (mm)", rainfall],
            ["Fertilizer (kg/ha)", fertilizer],
            ["Pesticide (kg/ha)", pesticide],
        ]
        table = Table(data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4CAF50")),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#D5F5E3")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BOTTOMPADDING', (0,0), (-1,0), 8)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Recommendations
        elements.append(Paragraph("Recommendations:", rec_heading_style))
        recs = [
            "Ensure optimal irrigation based on rainfall conditions.",
            "Use balanced fertilizer within recommended limits (100–300 kg/ha).",
            "Avoid overuse of pesticides; explore organic alternatives.",
            "Monitor soil health regularly and practice crop rotation.",
            "Use high-yield, climate-resilient seed varieties.",
            "Stay updated with government schemes and agricultural advisories."
        ]
        for r in recs: 
            elements.append(Paragraph(f"- {r}", normal_style))
        elements.append(Spacer(1, 12))

        # Future Outlook (Dynamic)
        elements.append(Paragraph("Future Outlook:", outlook_heading_style))
        for o in outlook:
            elements.append(Paragraph(f"- {o}", normal_style))
        elements.append(Spacer(1, 12))

        # Charts
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='PNG', bbox_inches='tight')
        img_buffer.seek(0)
        fig_width, fig_height = fig.get_size_inches()
        aspect_ratio = fig_height / fig_width
        img_width = 400
        img_height = img_width * aspect_ratio
        elements.append(Image(img_buffer, width=img_width, height=img_height))
        elements.append(Spacer(1, 12))

        # Resources
        elements.append(Paragraph("Useful Resources:", resource_heading_style))
        
        # Style for normal text
        normal_text_style = ParagraphStyle(
            'NormalTextStyle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.black
        )
        
        # Resources with clickable-looking URLs
        resources = [
            ("Indian Council of Agricultural Research (ICAR)", "https://icar.org.in/"),
            ("Ministry of Agriculture & Farmers Welfare", "https://agricoop.nic.in/"),
            ("FAO Crop Production Guidelines", "https://www.fao.org/crop-production/en/")
        ]
        
        for name, url in resources:
            resource_line = f"{name}: <u><font color='#1F618D'>{url}</font></u>"
            elements.append(Paragraph(resource_line, normal_text_style))
            elements.append(Spacer(1, 6))

        doc.build(elements)
        buffer.seek(0)

        st.download_button("⬇️ Download PDF Report", buffer, file_name="crop_yield_report.pdf", mime="application/pdf")


if __name__ == "__main__":
    main()
