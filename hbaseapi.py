import happybase

connection = happybase.Connection('localhost')


def create_table():
    connection.create_table(
        'cod',
        {'model_meta': dict(max_versions=10),
         'model': dict(max_versions=1, block_cache_enabled=False)
         }
    )


def get_models(user):
    table = connection.table('cod')
    row = table.row(user)
    return row[b'model:model']  # returns ML model


def insert_models():
    table = connection.table('cod')
    table.put('user1~model3', {b'model_meta:model_name': 'Model 3',b'model_meta:model_desc': 'User1 - Day 3 Model' ,
                         b'model_meta:model_created_on': '2701202021'})


def save_dict(row_key, state):
    table = connection.table('cod')
    table.put(row_key, {b'model:model': state})



insert_models()