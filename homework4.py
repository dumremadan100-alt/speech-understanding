def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    # Sort all birthday dates
    dates = sorted(birthdays.keys())

    # Find the first birthday after the given date
    for d in dates:
        if d > date:
            return d, birthdays[d]

    # If none are later, wrap around to the first birthday
    first = dates[0]
    return first, birthdays[first]