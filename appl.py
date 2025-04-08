import streamlit as st
from backend import QC # Assuming QC.py is in the same folder

st.set_page_config(page_title="Quantum Computing Library", layout="centered")
st.title("🧠 Quantum Computing Operations")

operation = st.selectbox(
    "Pick one operation to perform:",
    [
        "Matrix Multiplication (MM)",
        "Normalized Vector Calculation (normVec)",
        "Evolution of States Calculation (EOS)",
        "Is Unitary or Hermitian (MCheck)",
        "Eigenvalues and Eigenvectors (Eigs)",
        "Calculate Tensor Product of Two Matrices (tensor)",
        "Mini Quantum Circuit Simulation (QC)"
    ]
)

def parse_matrix_input(input_text):
    return [list(map(float, row.strip().split(','))) for row in input_text.strip().split('\n') if row]

def parse_vector_input(input_text):
    return list(map(float, input_text.strip().split(',')))

# Matrix Multiplication
if operation == "Matrix Multiplication (MM)":
    st.subheader("Matrix Multiplication")
    m1 = st.text_area("Enter Matrix 1 (comma-separated rows):", height=150)
    m2 = st.text_area("Enter Matrix 2 (comma-separated rows):", height=150)
    if st.button("Multiply"):
        try:
            mat1 = parse_matrix_input(m1)
            mat2 = parse_matrix_input(m2)
            result = QC.matrix_multiplication(mat1, mat2)
            st.write("Result:")
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")

# Normalized Vector
elif operation == "Normalized Vector Calculation (normVec)":
    st.subheader("Normalize a Vector")
    v = st.text_input("Enter vector (comma-separated):")
    if st.button("Normalize"):
        try:
            vec = parse_vector_input(v)
            result = QC.normalize_vector(vec)
            st.write("Normalized Vector:", result)
        except Exception as e:
            st.error(f"Error: {e}")

# Evolution of States
elif operation == "Evolution of States Calculation (EOS)":
    st.subheader("State Evolution: Matrix × Vector")
    m = st.text_area("Enter Matrix (comma-separated rows):", height=150)
    v = st.text_input("Enter Vector (comma-separated):")
    if st.button("Evolve"):
        try:
            mat = parse_matrix_input(m)
            vec = parse_vector_input(v)
            result = QC.evolve_states(mat, vec)
            st.write("Evolved State:", result)
        except Exception as e:
            st.error(f"Error: {e}")

# Check Unitary or Hermitian
elif operation == "Is Unitary or Hermitian (MCheck)":
    st.subheader("Check Matrix Type")
    m = st.text_area("Enter Matrix (comma-separated rows):", height=150)
    if st.button("Check"):
        try:
            mat = parse_matrix_input(m)
            result = QC.check_unitary_or_hermitian(mat)
            st.write("Matrix Check:", result)
        except Exception as e:
            st.error(f"Error: {e}")

# Eigenvalues and Eigenvectors
elif operation == "Eigenvalues and Eigenvectors (Eigs)":
    st.subheader("Calculate Eigenvalues and Eigenvectors")
    m = st.text_area("Enter Matrix (comma-separated rows):", height=150)
    if st.button("Calculate"):
        try:
            mat = parse_matrix_input(m)
            eigenvalues, eigenvectors = QC.get_eigenvalues_and_vectors(mat)
            st.write("Eigenvalues:", eigenvalues)
            st.write("Eigenvectors:", eigenvectors)
        except Exception as e:
            st.error(f"Error: {e}")

# Tensor Product
elif operation == "Calculate Tensor Product of Two Matrices (tensor)":
    st.subheader("Tensor Product of Two Matrices")
    m1 = st.text_area("Enter Matrix 1 (comma-separated rows):", height=150)
    m2 = st.text_area("Enter Matrix 2 (comma-separated rows):", height=150)
    if st.button("Compute Tensor Product"):
        try:
            mat1 = parse_matrix_input(m1)
            mat2 = parse_matrix_input(m2)
            result = QC.tensor_product(mat1, mat2)
            st.write("Tensor Product:", result)
        except Exception as e:
            st.error(f"Error: {e}")

# Mini Quantum Circuit
elif operation == "Mini Quantum Circuit Simulation (QC)":
    st.subheader("Mini Quantum Circuit")
    st.markdown("This will run a pre-defined quantum circuit simulation.")
    if st.button("Run Mini QC"):
        try:
            result = QC.run_mini_qc()
            st.write("Result:", result)
        except Exception as e:
            st.error(f"Error: {e}")
