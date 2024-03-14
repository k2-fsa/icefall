# Copyright      2024  Xiaomi Corp.        (authors: Zengrui Jin,)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def raise_grad_scale_is_too_small_error(cur_grad_scale: float):
    raise RuntimeError(
        f"""
        grad_scale is too small, exiting: {cur_grad_scale}

        ========================= NOTE =========================
        If you see this error, it means that the gradient scale is too small.

        The default base_lr is 0.045 / 0.05 (depends on which recipe you are 
        using), this is an empirical value obtained mostly using 4 * 32GB V100 
        GPUs with a max_duration of approx. 1,000. 
        The proper value of base_lr may vary depending on the number of GPUs 
        and the value of max-duration you are using. 

        To fix this issue, you may need to adjust the value of base_lr accordingly.

        We would suggest you to decrease the value of base_lr by 0.005 (e.g., 
        from 0.045 to 0.04), and try again. If the error still exists, you may 
        repeat the process until base_lr hits 0.02. (Note that this will lead to 
        certain loss of performance, but it should work. You can compensate this by
        increasing the num_epochs.)
        
        If the error still exists, you could try to seek help by raising an issue, 
        with a detailed description of (a) your computational resources, (b) the 
        base_lr and (c) the max_duration you are using, (d) detailed configuration 
        of your model.

        ========================================================
        """
    )
