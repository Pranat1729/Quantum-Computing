import numpy as np
import threading as td
import sympy as sp
import itertools
import sys
import os

#### Version 1.02
print(
    "--------------------------------------------------------------------------------------------"
)
venus_art = [
    "V       V  EEEEE  N     N  U     U  SSSSS",
    " V     V   E      NN    N  U     U  S     ",
    "  V   V    EEEE   N N   N  U     U  SSSSS ",
    "   V V     E      N  N  N  U     U      S  ",
    "    V      EEEEE  N   N N  UUUUUUU  SSSSS  ",
]

for line in venus_art:
    os.system("color 0b")
    print(line)
print(
    "--------------------------------------------------------------------------------------------"
)

### BY PRANAT

X = input(
    "Pick one operation to perform: \n Matrix Multiplication(MM),\n normalized vector calculation(normVec),\n Evolution of states calculation(EOS),\n Is Unitary or Hermitian(MCheck),\n Eigenvalues and eigenvectors(Eigs),\n Calculate tensor product of two matrices(tensor),\n mini QC(QC) \n>>"
)


class ComplexMatrixOp:

    def __init__(self, choice):
        self.choice = choice

    def check(self):
        if self.choice == "MM":
            self.exm()
        elif self.choice == "normVec":
            self.norm_Vec()
        elif self.choice == "EOS":
            self.EOS()
        elif self.choice == "MCheck":
            self.MCheck()
        elif self.choice == "Eigs":
            self.Eigenvalue_and_Eigenvector()
        elif self.choice == "tensor":
            self.tensor_product()
        elif self.choice == "QC":
            self.mini_QC_sim()
        else:
            print("I do not recognise these commands to access the above typed features type the words written inside parantheses.")

    @staticmethod
    def MM(A, B, n, m, p):
        result = [[0] * p for _ in range(n)]

        for i in range(n):
            for j in range(p):
                for k in range(m):
                    result[i][j] += A[i][k] * B[k][j]

        return result

    def exm(self):
        n, m = map(int, input("Enter the dimensions of matrix A (n x m): ").split())

        A = []
        print("Enter matrix A row by row:")
        for i in range(n):
            row = list(map(complex, input().split()))
            A.append(row)

        m2, p = map(int, input("Enter the dimensions of matrix B (m x p): ").split())

        if m != m2:
            print(
                "Error: The number of columns in A must equal the number of rows in B."
            )
        else:
            B = []
            print("Enter matrix B row by row:")
            for i in range(m2):
                row = list(map(complex, input().split()))
                B.append(row)

            result = list(ComplexMatrixOp.MM(A, B, n, m, p))

            print(result)

    def norm_Vec(self):
        n = int(input("Enter the length of the column/row vector"))
        N = []
        for i in range(n):
            a = complex(input("Enter the real part of the complex number: "))
            N.append(a)
        N_ = []
        for i in N:
            x = np.conjugate(i)
            N_.append(x)
        M = []
        for i in range(n):
            q = N_[i] * N[i]
            M.append(q)

        print(np.array(N) / (np.sqrt(sum(M))))

    def EOS(self):
        s = int(input("Enter the dimenson of your initial state vector: "))
        ini_state = []
        for i in range(s):
            a = complex(input("Enter the element at position " f"{i+1} :"))
            ini_state.append(a)

        Big = []
        n = int(input("Number of matrices you wanna multiply"))
        for i in range(n):
            p, m = map(
                int, input("Enter the dimensions of the matrix (n x m): ").split()
            )

            A = []
            print("Enter matrix row by row:")
            for j in range(p):
                row = list(map(complex, input().split()))
                A.append(row)
            Big.append(A)

        if len(Big) >= 2:
            X = np.matmul(Big[0], Big[1])
            for i in range(2, len(Big)):
                a = np.matmul(X, Big[i])
                X = a
            # print(X)
            print(np.matmul(X, ini_state))

        elif len(Big) == 1:
            print(np.matmul(Big[0], ini_state))

        else:
            print("Fuck you trynna do")

    def MCheck(self):

        def check_Hermitian(Big2):
            for big in Big2:
                x = np.transpose(big)
                for i in range(list(np.shape(x))[0]):
                    for j in range(list(np.shape(x))[1]):
                        x[i][j] = np.conjugate(x[i][j])

                if np.array_equal(x, big):
                    print("Hermitian Matrix")
                else:
                    print("It's not")

        def check_Unitarity(Big2):
            for big in Big2:
                x = np.transpose(big)
                for i in range(list(np.shape(x))[0]):
                    for j in range(list(np.shape(x))[1]):
                        x[i][j] = np.conjugate(x[i][j])

                a = np.matmul(x, big)

                if list(np.shape(a))[0] == list(np.shape(a))[1]:
                    if np.array_equal(a, np.identity(list(np.shape(a))[0])):
                        print("Unitary Matrix")
                    else:
                        print("NOT")
                else:
                    print("NOT")

        Big2 = []
        n = int(input("Number of matrices you wanna check the type of : "))
        for i in range(n):
            p, m = map(
                int, input("Enter the dimensions of the matrix (n x m): ").split()
            )

            A = []
            print("Enter matrix row by row:")
            for j in range(p):
                row = list(map(complex, input().split()))
                A.append(row)
            Big2.append(A)

        if len(Big2) >= 1:
            thread_1 = td.Thread(target=check_Hermitian(Big2))
            thread_2 = td.Thread(target=check_Unitarity(Big2))

            thread_1.start()
            thread_2.start()

            thread_1.join()
            thread_2.join()
        else:
            print("Fuck you trynna do")

    def Eigenvalue_and_Eigenvector(self):

        def eigenvectors(A, s):
            n = []
            for i in s:
                e1 = A - i * np.eye(np.shape(A)[0])
                n.append(e1)
            vec = []
            for i in n:
                x = sp.Matrix(i).nullspace()
                vec.append(x)
            print("Eigenvectors of the given operatore are: ", vec)

        def PLU(A):
            alpha = sp.symbols(" a")
            expr = A - alpha * np.eye(np.shape(A)[0])

            n = np.shape(expr)[0]
            L = sp.eye(n)
            U = sp.Matrix(expr).as_mutable()
            P = sp.eye(n)

            for i in range(n):
                max_row = i
                max_val = sp.simplify(abs(U[i, i]))

                for r in range(i + 1, n):
                    current_val = sp.simplify(abs(U[r, i]))
                    if current_val.is_number and max_val.is_number:
                        if current_val > max_val:
                            max_row, max_val = r, current_val
                    elif not max_val.is_number:
                        if sp.simplify(current_val - max_val).is_positive:
                            max_row, max_val = r, current_val

                if i != max_row:
                    U.row_swap(i, max_row)
                    P.row_swap(i, max_row)
                    if i > 0:
                        L.row_swap(i, max_row)

                for j in range(i + 1, n):
                    if U[i, i] == 0:
                        raise ValueError(
                            "Zero pivot encountered; LU decomposition cannot proceed."
                        )

                    factor = U[j, i] / U[i, i]
                    L[j, i] = factor
                    U.row_op(j, lambda v, k: v - factor * U[i, k])  # Row operation

            n = P.shape[0]
            identity = np.eye(n)
            swaps = 0

            P_copy = P.copy()

            for i in range(n):
                if not np.array_equal(
                    P_copy[i], identity[i]
                ):  # Row is not in correct position
                    # Find correct row position
                    for j in range(i + 1, n):
                        if np.array_equal(P_copy[j], identity[i]):
                            # Swap rows
                            P_copy[[i, j]] = P_copy[[j, i]]
                            swaps += 1
                            break

            det_A = ((-1) ** (swaps)) * (np.prod(np.diag(np.array(U))))
            s = sp.solve(det_A)
            print("Eigenvalues of the given operatore are: ", s)
            eigenvectors(A, s)

        Big3 = []
        n = int(
            input(
                "Number of matrices you wanna calculate eigenvalues and eigenvectors of : "
            )
        )
        for i in range(n):
            p, m = map(
                int, input("Enter the dimensions of the matrix (n x m): ").split()
            )

            A = []
            print("Enter matrix row by row:")
            for j in range(p):
                row = list(map(complex, input().split()))
                A.append(row)
            Big3.append(A)

        if len(Big3) >= 1:
            threads = []

            for big in Big3:
                thread = td.Thread(target=PLU, args=(big,))
                thread.start()
                threads.append(thread)

            for thread in threads:

                thread.join()
        else:
            print("fuck off")

    @staticmethod
    def mTP(A, B, n, m, m2, p):
        if (m == 1) and (p == 1):
            k = []
            for i in A:
                x = i[0] * B
                k.append(x)
            result = np.array(k).flatten()
            print(result)
            return result
        elif (n == 1) and (m2 == 1):
            k = []
            for i in A:
                for j in range(len(i)):
                    x = i[j] * np.array(B)
                    k.append(x)
            result = np.array(k).flatten()
            print(result.transpose())
            return result
        else:
            temp = []
            for i in range(n):
                for j in range(m):
                    x = A[i][j] * np.array(B)
                    temp.append(x)
            main = []
            for s in range(0, len(temp), m2):
                portion = temp[s : s + m2]
                hstacking = np.hstack(portion)
                main.append(hstacking)
            print(np.vstack(main))

    def tensor_product(self):
        n, m = map(int, input("Enter the dimensions of matrix A (n x m): ").split())

        A = []
        print("Enter matrix A row by row:")
        for i in range(n):
            row = list(map(complex, input().split()))
            A.append(row)

        m2, p = map(int, input("Enter the dimensions of matrix B (m x p): ").split())

        B = []
        print("Enter matrix B row by row:")
        for i in range(m2):
            row = list(map(complex, input().split()))
            B.append(row)
        ComplexMatrixOp.mTP(A, B, n, m, m2, p)

    def mini_QC_sim(self):
        N = int(input("Enter total no of qubit circuit you want to build(upto 4)>> "))
        qubits = []
        for i in range(N):
            x = list(map(float, input("Enter probabilities for qubit: ").split(",")))
            qubits.append(x)
            if x[0] + x[1] == 1:
                continue
            else:
                print("ERROR! Probabilities should add up to 1.")
                sys.exit()

        for i in range(len(qubits)):
            for j in range(len(qubits[i])):
                qubits[i][j] = np.sqrt(qubits[i][j])

        def phase_gate(a, theta):
            phase = np.array([[1, 0], [0, np.exp(j * np.radians(theta))]])
            res = ComplexMatrixOp.MM(phase, a, 2, 2, 1)
            return res

        def CNOT_gate(a):
            CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            res = ComplexMatrixOp.MM(CNOT, a, 2, 2, 1)
            return res

        def H_gate(a):
            H = np.array([[1, 1], [1, -1]])
            res = 1 / np.sqrt(2) * np.array((ComplexMatrixOp.MM(H, a, 2, 2, 1)))
            return res

        def decision(qubits):
            n = int(input("How many total gates do you want to apply?>> "))
            for i in range(n):
                x = input(
                    "which Gate do you want to apply? Hadamard(H)(1-qubit version), Controlled Not(CNOT)(2-qubit), Phase Gate(phase)(1-qubit)>> "
                )
                if x == "H":
                    y = list(
                        map(
                            int,
                            input(
                                "Which qubits do you want to apply this to? E.g, 1 and 1,2 for both, etc.>> "
                            ).split(","),
                        )
                    )
                    for i in range(len(y)):
                        a = H_gate(np.array(qubits[y[i] - 1]).reshape(-1, 1))
                        qubits[y[i] - 1] = a
                elif x == "CNOT":
                    y = list(
                        map(
                            int,
                            input(
                                "Which qubits do you want to apply this to? E.g, 1,2(2 being the target qubit. eg: control qubit, target qubit) etc.>> "
                            ).split(","),
                        )
                    )
                    b = ComplexMatrixOp.mTP(
                        np.array(qubits[y[0] - 1]).reshape(-1, 1),
                        np.array(qubits[y[1] - 1]).reshape(-1, 1),
                        2,
                        1,
                        2,
                        1,
                    )
                    qubits[y[1] - 1] = CNOT_gate(np.array(b).reshape(-1, 1))
                elif x == "phase":
                    y = list(
                        map(
                            int,
                            input(
                                "Which qubits do you want to apply this to? E.g, 1 and 1,2 for both, etc.>> "
                            ).split(","),
                        )
                    )
                    for i in range(len(y)):
                        theta = int(input("Enter the phase factor in degrees>> "))
                        a = phase_gate(np.array(qubits[y[i] - 1]).reshape(-1, 1), theta)
                        qubits[y[i] - 1] = a
                else:
                    print("Type properly.")
            print(qubits)

        decision(qubits)


choice = X
O1 = ComplexMatrixOp(choice)
O1.check()
