import unittest

# Python3 code to Check for
# balanced parentheses in an expression
open_list = ["[", "{", "("]
close_list = ["]", "}", ")"]


# Function to check parentheses
def check(myStr):
    stack = []
    for i in myStr:
        if i in open_list:
            stack.append(i)
        elif i in close_list:
            pos = close_list.index(i)
            if ((len(stack) > 0) and
                    (open_list[pos] == stack[len(stack) - 1])):
                stack.pop()
            else:
                return "Unbalanced"
    if len(stack) == 0:
        return "Balanced"
    else:
        return "Unbalanced"


# Driver code
# string = "{[]{()}}"
# assert (check(string) == 'Balanced')
# print(string, "-", check(string))
#
# string = "[{}{})(]"
# assert (check(string) == 'Balanced')
# print(string, "-", check(string))
#
# string = "((()"
# assert (check(string) == 'Balanced')
# print(string, "-", check(string))
#
# string = "((()"
# assert (check(string) == 'Balanced')
# print(string, "-", check(string))

class TestFunc(unittest.TestCase):

    def test1(self):
        string = "{[]{()}}"
        self.assertEqual(check(string), "Balanced")

    def test2(self):
        string = "[{}{})(]"
        self.assertEqual(check(string), "Balanced")

    def test3(self):
        string = "hello"
        self.assertIsNone(string)


if __name__ == '__main__':
    unittest.main()
