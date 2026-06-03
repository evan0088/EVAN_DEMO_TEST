if __name__ == '__main__':
    list1 = [1,0,1,3]
    list2 = [2,0,2,5]

    list3 = []

    list3.append(list1)
    list3.append(list2)

    list4 = []
    list4.extend(list1)
    list4.extend(list2)

    print(f'{list3}   \t  {len(list3)}')
    print(f'{list4} \t  {len(list4)}')