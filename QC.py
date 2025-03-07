import numpy as np 
import threading as td
import sympy as sp
import itertools

X = input("Pick one operation to perform: \n Matrix Multiplication(MM),\n normalized vector calculation(normVec),\n Evolution of states calculation(EOS),\n Is Unitary or Hermitian(MCheck),\n Eigenvalues and eigenvectors(Eigs),\n Calculate tensor product of two matrices(tensor)\n >>")


### BY PRANAT 


class ComplexMatrixOp():

	def __init__(self,choice):
		self.choice = choice

	def check(self):
		if self.choice == 'MM':
			self.exm()
		elif self.choice == 'normVec':
			self.norm_Vec()
		elif self.choice == 'EOS':
			self.EOS()
		elif self.choice == 'MCheck':
			self.MCheck()
		elif self.choice == 'Eigs':
			self.Eigenvalue_and_Eigenvector()
		elif self.choice == 'tensor':
			self.tensor_product()
		else:
			print("I raise you a fuck you error.Type properly")

	def MM(self,A, B, n, m, p):
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
			print("Error: The number of columns in A must equal the number of rows in B.")
		else:
			B = []
			print("Enter matrix B row by row:")
			for i in range(m2):
				row = list(map(complex, input().split()))
				B.append(row)

			result = list(self.MM(A, B, n, m, p))

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
			q = N_[i]*N[i]
			M.append(q)

		print(np.array(N)/(np.sqrt(sum(M))))

	def EOS(self):
		s = int(input("Enter the dimenson of your initial state vector: "))
		ini_state = []
		for i in range(s):
			a = complex(input("Enter the element at position " f"{i+1} :"))
			ini_state.append(a)

		Big = []
		n = int(input("Number of matrices you wanna multiply"))
		for i in range(n):
			p, m = map(int, input("Enter the dimensions of the matrix (n x m): ").split())

			A = []
			print("Enter matrix row by row:")
			for j in range(p):
				row = list(map(complex, input().split()))
				A.append(row)
			Big.append(A)

		if len(Big)>=2:
			X = np.matmul(Big[0],Big[1])
			for i in range(2,len(Big)):
				a = np.matmul(X,Big[i])
				X = a
			#print(X)
			print(np.matmul(X,ini_state))

		elif len(Big)==1:
			print(np.matmul(Big[0],ini_state))

		else:
			print("Fuck you trynna do")

	def MCheck(self):

		def check_Hermitian(Big2):
			for big in Big2:
				x = np.transpose(big)
				for i in range(list(np.shape(x))[0]):
					for j in range(list(np.shape(x))[1]):
						x[i][j] = np.conjugate(x[i][j])

				if np.array_equal(x,big):
					print("Hermitian Matrix")
				else:
					print("It's not")

		def check_Unitarity(Big2):
			for big in Big2:
				x = np.transpose(big)
				for i in range(list(np.shape(x))[0]):
					for j in range(list(np.shape(x))[1]):
						x[i][j] = np.conjugate(x[i][j])

				a = np.matmul(x,big)

				if (list(np.shape(a))[0] == list(np.shape(a))[1]):
					if np.array_equal(a, np.identity(list(np.shape(a))[0])):
						print("Unitary Matrix")
					else:
						print("NOT")
				else:
					print("NOT")

		Big2 = []
		n = int(input("Number of matrices you wanna check the type of : "))
		for i in range(n):
			p, m = map(int, input("Enter the dimensions of the matrix (n x m): ").split())

			A = []
			print("Enter matrix row by row:")
			for j in range(p):
				row = list(map(complex, input().split()))
				A.append(row)
			Big2.append(A)

		if len(Big2)>=1:
			thread_1 = td.Thread(target=check_Hermitian(Big2))
			thread_2 = td.Thread(target=check_Unitarity(Big2))

			thread_1.start()
			thread_2.start()

			thread_1.join()
			thread_2.join()
		else:
			print("Fuck you trynna do")


	def Eigenvalue_and_Eigenvector(self):

		def eigenvalues(A):
			eigenvalues, eigenvectors = np.linalg.eig(A)
			print("Eigenvalues:", eigenvalues)
			print("Eigenvectors:")
			print(eigenvectors)

			#alpha = sp.symbols('a')
			#e1 = alpha*np.identity(list(np.shape(A))[0])
			#r = sp.Eq(np.linalg.det(np.array(A)-e1),0)

			#sol = sp.solve(r, alpha)
			#print(sol)
		Big3 = []
		n = int(input("Number of matrices you wanna calculate eigenvalues and eigenvectors of : "))
		for i in range(n):
			p, m = map(int, input("Enter the dimensions of the matrix (n x m): ").split())

			A = []
			print("Enter matrix row by row:")
			for j in range(p):
				row = list(map(complex, input().split()))
				A.append(row)
			Big3.append(A)

		if len(Big3)>=1:
			threads = []

			for big in Big3:
				thread = td.Thread(target=eigenvalues,args=(big,))
				thread.start()
				threads.append(thread)

			for thread in threads:

				thread.join()
		else:
			print('fuck off')


	def tensor_product(self):

		def mTP(A,B,n,m,m2,p):
			temp = []
			for i in range(n):
				for j in range(m):
						x = A[i][j]*np.array(B)
						temp.append(x)
			main = []
			for s in range(0,len(temp),m2):
				portion = temp[s:s+m2]
				hstacking = np.hstack(portion)
				main.append(hstacking)

			print(np.vstack(main))

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
		mTP(A,B,n,m,m2,p)	

### Incorporate streamlit bot, will use funcs like st.text_input(), etc. Generate a message prompt and something bit interactive.
choice = X
O1 = ComplexMatrixOp(choice)
O1.check()