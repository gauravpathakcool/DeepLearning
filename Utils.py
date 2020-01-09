arry = ["6", "54", "326", "48","12","12345","9","8","7","0","99"]

if __name__ == "__main__":
    print("hello")


def findlen():
    i = 0
    length = 0
    while i < len(arry):
        element = arry[i]
        cur_len = len(element)
        if cur_len > length:
            length = cur_len
        i = i + 1
    return length


def modifyelement(length):
    j = 0
    flag = 0
    modify_dict = {}
    while j < len(arry):
        ele = arry[j]
        if len(ele) < length:
            temp = length - len(ele)
            while flag < temp:
                ele = ele + "0"
                flag = flag + 1
        modify_dict.update({arry[j]: ele})
        flag = 0
        j = j + 1
    return modify_dict


def SortAndDisplay(dict):
    number = ""
    res = sorted(dict.items(), reverse=True)
    for item in res:
        number = number + str(item[0])
    return number


length = findlen()
dict = modifyelement(length)
result = SortAndDisplay(dict)
print(result)
