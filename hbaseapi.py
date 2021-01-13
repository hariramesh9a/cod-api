import happybase

connection = happybase.Connection('localhost:2181')


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
