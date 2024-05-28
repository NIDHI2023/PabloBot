from news import get_title

def get_response(user_input: str) -> str:
    if (user_input == 'Hi'):
        return 'Hello!'
    elif (user_input == 'news'):
        return get_title()
    else:
        return 'bozo'