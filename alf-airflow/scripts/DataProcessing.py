from datetime import datetime

class PreTrainingLayer:
    def __init__(self):
        pass

    @staticmethod
    def format_value(value):
        if isinstance(value, list):
            return ', '.join(str(item) for item in value)
        return str(value)

    def format_sample_to_text(self, sample):
        text_parts = []
        timestamp_value = None

        for field, value in sample.items():
            if field.lower() == 'label':
                continue

            if field.lower() == 'timestamps':
                timestamp_value = value
                continue

            if field.lower() == '@timestamp':
                dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
                timestamp_value = [dt.timestamp()]
                continue

            if isinstance(value, dict):
                sub_parts = []
                for sub_field, sub_value in value.items():
                    if sub_field.lower() == 'label':
                        continue
                    if sub_field.lower() == 'timestamps':
                        timestamp_value = sub_value
                        continue

                    formatted_sub_value = self.format_value(sub_value)
                    sub_parts.append(f"[{sub_field}] {formatted_sub_value}")
                text_parts.append(f"[{field}] {{{' '.join(sub_parts)}}}")
            else:
                formatted_value = self.format_value(value)
                text_parts.append(f"[{field}] {formatted_value}")
        if timestamp_value is None:
            print('exit no timestamp')

        return ' '.join(text_parts), timestamp_value