import psycopg2
import copy
import random


def get_columns(train_val_metrics, test_val_metrics, type_dict=None):
    if type_dict is None:
        type_dict = {}

    columns = {}
    for name in train_val_metrics:
        columns['train_' + name] = "double precision" if name not in type_dict else type_dict[name]

    for name in test_val_metrics:
        columns['test_' + name] = "double precision" if name not in type_dict else type_dict[name]
    return columns


def get_primary_key_and_value(key_type_value):
    primary_value = {}
    primary_key = {}
    for key, v in key_type_value.items():
        sql_type, value = v
        primary_key[key] = sql_type
        if value is not None:
            if isinstance(value, list) or isinstance(value, tuple):
                first_flag = True
                list_str = ""
                for i in value:
                    if isinstance(i, str):
                        i = '"' + i + '"'
                        list_str += i if first_flag else ',' + i
                    else:
                        list_str += str(i) if first_flag else ',' + str(i)
                    first_flag = False
                value = "{" + list_str + "}"
            primary_value[key] = value
    return primary_key, primary_value


class PathProjector:
    def __init__(self, database, table_name='gan_model_path_projector'):
        self.database = database
        self.table_name = table_name

    def connect(self):
        self.conn = psycopg2.connect(**self.database)

    def init(self):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute("CREATE TABLE if not exists " + self.table_name +
                       "(" + 'model_path text unique, path_id integer unique' + ");")
        self.conn.commit()

    def get_path_id(self, path):
        cursor = self.conn.cursor()
        cursor.execute('select path_id from ' + self.table_name + " where model_path = '" + path + "';")
        rows = cursor.fetchall()
        for i in rows:
            return i[0]
        return None

    def set_path_id(self, path):
        cursor = self.conn.cursor()
        cursor.execute('select path_id from ' + self.table_name + ';')
        rows = cursor.fetchall()
        ids = tuple(rows)
        if ids:
            ids = next(zip(*ids))
        # print(ids)
        new_id = random.randint(0, 2147483647)
        while new_id in ids:
            new_id = random.randint(0, 2147483647)
        cursor.execute('insert into ' + self.table_name + " (model_path, path_id) values('{}', {})".format(path, new_id))
        self.conn.commit()
        return str(new_id)

    @staticmethod
    def dict2path(table_name, parameters):
        return table_name + str(dict(sorted(parameters.items(), key=lambda x: x[0], reverse=False))).replace("'", '"')

    def close(self):
        self.conn.close()


class WriteToDatabase:
    def __init__(self, database, table_name, primary_keys, other_keys, kept_params, refresh_params, refresh=False, overwrite=False):
        self.database = database
        self.table_name = table_name
        self.primary_keys = primary_keys
        self.other_keys = other_keys
        self.kept_params = kept_params
        self._refresh_params = refresh_params
        self.is_refresh = refresh
        self.overwrite = overwrite

    def init_conn(self):
        keepalive_kwargs = {
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 5,
            "keepalives_count": 5,
        }
        self.conn = psycopg2.connect(**self.database, **keepalive_kwargs)

    def init(self):
        primary_params = self.primary_keys
        other_params = self.other_keys
        self.init_conn()
        cursor = self.conn.cursor()
        t_str = ""
        for key, t in primary_params.items():
            t_str += "{} {},\n".format(key, t)
        for key, t in other_params.items():
            t_str += "{} {},\n".format(key, t)
        pk = list(primary_params.keys())
        pk_str_ = pk[0]
        for i in pk[1:]:
            pk_str_ = pk_str_ + " ," + i
        pk_str = "CONSTRAINT pk_{} PRIMARY KEY ".format(
            self.table_name) + "(" + pk_str_ + ")"

        # print("CREATE TABLE if not exists " + self.table_name +
        #       "(" + t_str + pk_str + ");")

        cursor.execute("CREATE TABLE if not exists " + self.table_name +
                       "(" + t_str + pk_str + ");")
        print(self.is_refresh)
        if self.is_refresh:
            self.refresh()
        self.conn.commit()

    def get_identity_dict(self, identity_dict):
        identity_dict_ = copy.deepcopy(self.kept_params)
        for key, value in identity_dict_.items():
            if isinstance(value, str):
                value = "'" + value + "'"
            identity_dict_[key] = value

        for key, value in identity_dict.items():
            if isinstance(value, str):
                value = "'" + value + "'"
            identity_dict_[key] = str(value)

        return copy.deepcopy(identity_dict_)

    @property
    def refresh_params(self):
        refresh_params = copy.deepcopy(self._refresh_params)
        for key, value in refresh_params.items():
            if isinstance(value, str):
                value = "'" + value + "'"
            refresh_params[key] = value
        return refresh_params

    def write(self, identity_dict, dict_write):
        cursor = self.conn.cursor()

        dict_write_ = self.get_identity_dict(identity_dict)
        for key, value in dict_write.items():
            if isinstance(value, str):
                value = "'" + value + "'"
            dict_write_[key] = str(value)

        insert_str = "("
        for key in dict_write_:
            insert_str += key + ','
        insert_str = insert_str[:-1] + ') values ('
        for value in dict_write_.values():
            insert_str += str(value) + ','
        insert_str = insert_str[:-1] + ')'
        update_str = "update set " + self.equal_concat(
            dict_write, ',')
        if self.overwrite:
            cursor.execute("insert into {} ".format(
                self.table_name) + insert_str + " on conflict on CONSTRAINT pk" + "_" + self.table_name + " do {};".format(update_str))
        else:
            cursor.execute("insert into {} ".format(
                self.table_name) + insert_str + ";")
        self.conn.commit()

    def equal_concat(self, condition_dict, concat='and'):
        condition = ""
        first = True
        for key, value in condition_dict.items():
            if not first:
                condition = condition + concat + \
                    " {} = {} ".format(key, value)
            else:
                condition = condition + " {} = {} ".format(key, value)
                first = False
        return condition

    def refresh(self):
        cursor = self.conn.cursor()
        del_str = "delete from " + self.table_name + " where "
        del_str += self.equal_concat(self.refresh_params) + ";"
        # print(del_str)
        # exit()
        cursor.execute(del_str)
        # self.conn.commit()

    def close(self):
        self.conn.close()


class WriteToDatabaseWithPath(WriteToDatabase):
    def __init__(self, database, table_name, primary_keys, other_keys, kept_params, refresh_params, refresh=False, overwrite=False, path_column_name="path_id"):
        super().__init__(database, table_name, primary_keys, other_keys, kept_params, refresh_params, refresh, overwrite)
        self.path_column_name = path_column_name
        self._delete_default = False
        self.other_keys[self.path_column_name] = 'integer'

    def write(self, identity_dict, dict_write):
        if self._delete_default:
            cursor = self.conn.cursor()
            identity_dict_ = self.get_identity_dict(self._path_identity_dict)
            cursor.execute("delete from " + self.table_name + " where " + self.equal_concat(identity_dict_) + ";")
            self._delete_default = False
        dict_write[self.path_column_name] = self.path_id
        super().write(identity_dict, dict_write)

    def write_path_id(self, identity_dict, dict_write):
        path_id = self._get_path_id()
        self._path_identity_dict = identity_dict
        identity_dict_ = self.get_identity_dict(identity_dict)
        cursor = self.conn.cursor()
        cursor.execute("select path_id from " + self.table_name + " where " + self.equal_concat(identity_dict_) + ";")
        if cursor.fetchall():
            self._delete_default = True
        if path_id is None:
            cursor.execute("begin;")
            cursor.execute("lock table " + self.table_name + " in ACCESS EXCLUSIVE MODE;")
            path_id = self._gen_new_path_id()
            dict_write[self.path_column_name] = path_id
            super().write(identity_dict, dict_write)
            self._delete_default = True
        else:
            self.conn.commit()
        self.path_id = path_id

    def get_path_id(self):
        return self.path_id

    def _get_path_id(self):
        cursor = self.conn.cursor()
        kept_params = self.get_identity_dict(dict())
        cursor.execute('select ' + self.path_column_name + ' from ' + self.table_name + " where " + self.equal_concat(kept_params) + ";")
        rows = cursor.fetchall()
        for i in rows:
            return i[0]
        return None

    def _gen_new_path_id(self):
        cursor = self.conn.cursor()
        cursor.execute('select distinct ' + self.path_column_name + ' from ' + self.table_name + ';')
        rows = cursor.fetchall()
        ids = tuple(rows)
        if ids:
            ids = next(zip(*ids))
        # print(ids)
        new_id = random.randint(0, 2147483647)
        while new_id in ids:
            new_id = random.randint(0, 2147483647)
        return new_id


