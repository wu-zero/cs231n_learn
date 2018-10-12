class Matrix:
    def __init__(self,data_list):
        self.data = data_list
        self.row = len(data_list)
        self.col = len(data_list[0])
        self.shape = (self.row, self.col)

    def __str__(self):
        result = ''
        for i in range(self.row):
            for j in range(self.col):
                result = result + str(self.data[i][j])+' '
            result = result + '\n'
        return result

    def sum(self,matrix2):
        if self.shape != matrix2.shape:
            raise Exception("两个矩阵大小不同")
        else:
            result = []
            for i in range(self.row):
                result_row = []
                for j in range(self.col):
                    result_row.append(self.data[i][j]+matrix2.data[i][j])
                result.append(result_row)
            #print(result)
            return Matrix(result)
    # dot product
    def dot(self,matrix2):
        if self.col != matrix2.row:
            raise Exception("前者的列不等于后者的行，无法计算")
        else:
            result = []
            for i in range(self.row):
                result_row = []
                for j in range(self.col):
                    result_ij = 0
                    for k in range(self.col):
                        result_ij = result_ij + self.data[i][k]*matrix2.data[k][j]
                    result_row.append(result_ij)
                result.append(result_row)
            #print(result)
            return Matrix(result)







if __name__ == '__main__':
    a = [[1,2,3],
         [4,5,6]]
    b = [[2,3,4],
         [1,1,1]]
    c = [[1,2,3],
         [1,1,1],
         [1,2,2]]

    a_matrix = Matrix(a)
    b_matrix = Matrix(b)
    c_matrix = Matrix(c)

    a_matrix.sum(b_matrix)
    a_matrix.dot(c_matrix)
    print('a=',a_matrix)
    print('b=',b_matrix)
    print('c=',c_matrix)
    print('a.dot(c)=',a_matrix.dot(c_matrix))