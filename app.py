import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Your Name - Professional CV",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
    .contact-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .skill-bar {
        height: 12px;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        border-radius: 6px;
        margin: 2px 0px;
    }
    .project-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER SECTION =====
col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://via.placeholder.com/150x150/1f77b4/FFFFFF?text=Photo", 
             width=150, 
             caption="Your Name")

with col2:
    st.markdown('<h1 class="main-header">YOUR FULL NAME</h1>', unsafe_allow_html=True)
    st.markdown("### Python Developer & Data Scientist")
    st.markdown("""
    Passionate developer with 5+ years of experience building scalable applications. 
    Specialized in Python, data analysis, and backend development. Strong problem-solver 
    with a track record of delivering high-impact solutions.
    """)

# ===== CONTACT INFORMATION =====
st.markdown("---")
st.markdown('<h2 class="section-header">📞 Contact Information</h2>', unsafe_allow_html=True)

contact_cols = st.columns(4)
with contact_cols[0]:
    st.markdown("**📧 Email**")
    st.write("your.email@example.com")
with contact_cols[1]:
    st.markdown("**📱 Phone**")
    st.write("+1 (555) 123-4567")
with contact_cols[2]:
    st.markdown("**💼 LinkedIn**")
    st.write("[linkedin.com/in/yourname](https://linkedin.com)")
with contact_cols[3]:
    st.markdown("**🐙 GitHub**")
    st.write("[github.com/yourusername](https://github.com)")

# ===== TECHNICAL SKILLS =====
st.markdown("---")
st.markdown('<h2 class="section-header">🛠️ Technical Skills</h2>', unsafe_allow_html=True)

# Programming Languages
st.subheader("Programming Languages")
skills = {
    'Python': 95,
    'SQL': 88,
    'JavaScript': 75,
    'Bash/Shell': 80,
    'R': 70
}

for skill, level in skills.items():
    st.write(f"**{skill}**")
    st.markdown(f'<div class="skill-bar" style="width: {level}%"></div>', unsafe_allow_html=True)

# Technologies
col1, col2 = st.columns(2)
with col1:
    st.subheader("Frameworks & Libraries")
    st.write("""
    • FastAPI • Django • Flask
    • Pandas • NumPy • Scikit-learn
    • TensorFlow • Streamlit
    • Plotly • Matplotlib
    """)

with col2:
    st.subheader("Tools & Platforms")
    st.write("""
    • Docker • Git • AWS
    • PostgreSQL • MySQL
    • Linux • Jupyter
    • VS Code • PyCharm
    """)

# ===== PROFESSIONAL EXPERIENCE =====
st.markdown("---")
st.markdown('<h2 class="section-header">💼 Professional Experience</h2>', unsafe_allow_html=True)

# Job 1
st.subheader("Senior Python Developer | Tech Solutions Inc.")
st.write("**June 2020 - Present | San Francisco, CA**")
st.write("""
• Led development of data processing platform handling 1M+ daily users
• Improved system performance by 60% through optimization
• Mentored 3 junior developers and established coding standards
• Technologies: Python, FastAPI, PostgreSQL, AWS, Docker
""")

# Job 2
st.subheader("Python Developer | Data Analytics Corp")
st.write("**January 2018 - May 2020 | New York, NY**")
st.write("""
• Developed ETL pipelines processing 10TB+ of data monthly
• Built dashboards that provided insights leading to 20% cost reduction
• Automated reporting processes saving 15+ hours weekly
• Technologies: Python, Pandas, Django, MySQL
""")

# ===== EDUCATION =====
st.markdown("---")
st.markdown('<h2 class="section-header">🎓 Education</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Master of Science in Computer Science")
    st.write("**Stanford University**")
    st.write("2016 - 2018 | GPA: 3.8/4.0")
    st.write("*Thesis: Advanced Machine Learning Applications*")

with col2:
    st.subheader("Bachelor of Science in Software Engineering")
    st.write("**UC Berkeley**")
    st.write("2012 - 2016 | GPA: 3.7/4.0")
    st.write("*Graduated Magna Cum Laude*")

# ===== PROJECTS =====
st.markdown("---")
st.markdown('<h2 class="section-header">🚀 Key Projects</h2>', unsafe_allow_html=True)

# Project 1
with st.container():
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.subheader("E-Commerce Analytics Platform")
    st.write("""
    Built a comprehensive analytics platform processing 10,000+ daily transactions. 
    Implemented real-time dashboards and predictive models with 92% accuracy.
    """)
    st.write("**Technologies:** Python, FastAPI, Pandas, React, PostgreSQL")
    st.markdown('</div>', unsafe_allow_html=True)

# Project 2
with st.container():
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.subheader("Flood Risk Analysis System")
    st.write("""
    Developed environmental data analysis system for flood prediction. 
    Processed sensor data and implemented risk assessment algorithms.
    """)
    st.write("**Technologies:** Python, NumPy, Pandas, Streamlit")
    st.markdown("[View Project](https://python-portfolio-blavg9oswmrethpkgqvmdw.streamlit.app/)")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== CERTIFICATIONS =====
st.markdown("---")
st.markdown('<h2 class="section-header">🏆 Certifications</h2>', unsafe_allow_html=True)

cert_cols = st.columns(3)
with cert_cols[0]:
    st.write("**AWS Certified Developer**")
    st.write("Amazon Web Services | 2023")
with cert_cols[1]:
    st.write("**Python for Data Science**")
    st.write("Coursera | 2022")
with cert_cols[2]:
    st.write("**Machine Learning Specialist**")
    st.write("Stanford Online | 2021")

# ===== DOWNLOAD SECTION =====
st.markdown("---")
st.markdown('<h2 class="section-header">📥 Get in Touch</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.write("**Available for:**")
    st.write("• Full-time positions")
    st.write("• Contract work")
    st.write("• Technical consulting")

with col2:
    st.write("**Open to roles:**")
    st.write("• Senior Python Developer")
    st.write("• Data Scientist")
    st.write("• Backend Engineer")

# Generate PDF button (conceptual)
if st.button("📄 Generate PDF Version"):
    st.info("PDF generation would be implemented here for a downloadable resume")

# ===== FOOTER =====
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>
        Built with Python & Streamlit • 
        Last updated: {datetime.now().strftime("%B %d, %Y")} • 
        <a href="https://python-portfolio-blavg9oswmrethpkgqvmdw.streamlit.app/">View My Technical Portfolio</a>
    </p>
</div>
""", unsafe_allow_html=True)
