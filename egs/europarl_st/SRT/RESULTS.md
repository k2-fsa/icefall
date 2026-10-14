# Detailed Results

This file contains the complete per-direction results for all 72 translation directions on Europarl-ST.

## Many-to-Many Joint Training (All Directions)

### WER
<table>
  <thead>
    <tr>
      <th rowspan="2">SRC\TGT</th>
      <th rowspan="2" align="left">Model</th>
      <th colspan="9">WER (%) &#8595;</th>
    </tr>
    <tr>
      <th>de</th><th>en</th><th>es</th><th>fr</th><th>it</th><th>nl</th><th>pl</th><th>pt</th><th>ro</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <th rowspan="7">de</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>-</td><td>21.80</td><td>26.64</td><td>27.12</td><td>27.44</td><td>26.43</td><td>26.51</td><td>26.51</td><td>26.62</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>-</td><td>19.09</td><td>18.77</td><td>18.82</td><td>19.09</td><td>18.86</td><td>18.79</td><td>18.99</td><td>18.86</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td>-</td><td><strong>18.01</strong></td><td><strong>17.84</strong></td><td><strong>17.85</strong></td><td>18.23</td><td><strong>17.92</strong></td><td><strong>17.75</strong></td><td><strong>17.93</strong></td><td>17.99</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>-</td><td>18.83</td><td>18.76</td><td>18.75</td><td>18.92</td><td>18.74</td><td>18.61</td><td>18.81</td><td>18.85</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>-</td><td>18.13</td><td>17.88</td><td>17.86</td><td><strong>18.16</strong></td><td><strong>17.92</strong></td><td>17.80</td><td><strong>17.93</strong></td><td><strong>17.97</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>-</td><td>18.83</td><td>18.59</td><td>18.62</td><td>18.95</td><td>18.60</td><td>18.44</td><td>18.57</td><td>18.78</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>-</td><td>18.66</td><td>18.35</td><td>18.33</td><td>18.65</td><td>18.41</td><td>18.30</td><td>18.51</td><td>18.50</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">en</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>16.42</td><td>-</td><td>17.32</td><td>17.56</td><td>17.08</td><td>17.45</td><td>17.29</td><td>17.18</td><td>17.21</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>13.92</td><td>-</td><td>13.94</td><td>14.02</td><td>13.79</td><td>13.84</td><td>13.99</td><td>13.90</td><td>13.53</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>12.94</strong></td><td>-</td><td><strong>12.93</strong></td><td><strong>13.02</strong></td><td><strong>12.84</strong></td><td><strong>12.87</strong></td><td><strong>12.92</strong></td><td><strong>12.97</strong></td><td><strong>12.64</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>13.28</td><td>-</td><td>13.27</td><td>13.33</td><td>13.14</td><td>13.18</td><td>13.31</td><td>13.34</td><td>12.88</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>13.26</td><td>-</td><td>13.25</td><td>13.35</td><td>13.05</td><td>13.16</td><td>13.26</td><td>13.25</td><td>12.94</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>13.41</td><td>-</td><td>13.38</td><td>13.45</td><td>13.26</td><td>13.31</td><td>13.43</td><td>13.39</td><td>13.04</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>13.50</td><td>-</td><td>13.49</td><td>13.61</td><td>13.46</td><td>13.39</td><td>13.53</td><td>13.48</td><td>13.13</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">es</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>21.29</td><td>17.77</td><td>-</td><td>22.25</td><td>22.83</td><td>22.68</td><td>21.93</td><td>22.82</td><td>22.66</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>15.96</td><td>15.80</td><td>-</td><td>15.91</td><td>15.69</td><td>15.97</td><td>15.85</td><td>15.89</td><td>15.75</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td>15.30</td><td>15.14</td><td>-</td><td>15.27</td><td><strong>15.02</strong></td><td>15.31</td><td>15.26</td><td>15.25</td><td>15.14</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>15.61</td><td>15.51</td><td>-</td><td>15.65</td><td>15.41</td><td>15.60</td><td>15.56</td><td>15.52</td><td>15.57</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td><strong>15.19</strong></td><td><strong>15.13</strong></td><td>-</td><td><strong>15.21</strong></td><td>15.03</td><td><strong>15.21</strong></td><td><strong>15.14</strong></td><td><strong>15.11</strong></td><td><strong>14.89</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>15.96</td><td>15.81</td><td>-</td><td>15.97</td><td>15.74</td><td>15.99</td><td>15.89</td><td>15.90</td><td>15.82</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>15.46</td><td>15.22</td><td>-</td><td>15.41</td><td>15.16</td><td>15.46</td><td>15.34</td><td>15.35</td><td>15.11</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">fr</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>19.37</td><td>16.07</td><td>19.82</td><td>-</td><td>20.41</td><td>19.30</td><td>19.45</td><td>19.86</td><td>20.80</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>13.40</td><td>13.38</td><td>13.28</td><td>-</td><td>13.42</td><td>13.36</td><td>13.39</td><td>13.38</td><td>13.37</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td>12.58</td><td>12.51</td><td>12.53</td><td>-</td><td>12.51</td><td>12.50</td><td>12.56</td><td>12.55</td><td>12.65</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>13.15</td><td>13.10</td><td>13.02</td><td>-</td><td>13.08</td><td>12.99</td><td>13.15</td><td>13.16</td><td>13.27</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td><strong>12.53</strong></td><td><strong>12.49</strong></td><td><strong>12.37</strong></td><td>-</td><td><strong>12.47</strong></td><td><strong>12.45</strong></td><td><strong>12.48</strong></td><td><strong>12.52</strong></td><td><strong>12.51</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>12.75</td><td>12.77</td><td>12.62</td><td>-</td><td>12.70</td><td>12.67</td><td>12.69</td><td>12.65</td><td>12.78</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>12.58</td><td>12.64</td><td>12.49</td><td>-</td><td>12.55</td><td>12.65</td><td>12.56</td><td>12.64</td><td>12.74</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">it</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>18.18</td><td>15.05</td><td>19.19</td><td>19.32</td><td>-</td><td>19.06</td><td>18.60</td><td>19.00</td><td>19.91</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>13.10</td><td>13.19</td><td>13.13</td><td>13.24</td><td>-</td><td>13.17</td><td>12.98</td><td>13.18</td><td>13.27</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>12.50</strong></td><td><strong>12.41</strong></td><td><strong>12.52</strong></td><td><strong>12.63</strong></td><td>-</td><td><strong>12.59</strong></td><td><strong>12.42</strong></td><td><strong>12.62</strong></td><td><strong>12.66</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>13.00</td><td>12.92</td><td>13.03</td><td>13.04</td><td>-</td><td>13.05</td><td>12.89</td><td>13.11</td><td>13.27</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>12.67</td><td>12.63</td><td>12.77</td><td>12.80</td><td>-</td><td>12.74</td><td>12.67</td><td>12.84</td><td>12.95</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>12.91</td><td>12.91</td><td>12.97</td><td>13.03</td><td>-</td><td>12.97</td><td>12.85</td><td>13.13</td><td>13.08</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>12.92</td><td>12.90</td><td>12.96</td><td>13.07</td><td>-</td><td>13.07</td><td>12.84</td><td>13.12</td><td>13.13</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">nl</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>38.99</td><td>32.95</td><td>38.85</td><td>38.85</td><td>39.52</td><td>-</td><td>38.99</td><td>39.32</td><td>39.26</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>28.59</td><td>28.65</td><td>28.73</td><td>28.46</td><td>28.62</td><td>-</td><td>28.46</td><td>28.46</td><td>28.47</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>27.01</strong></td><td><strong>27.23</strong></td><td><strong>26.89</strong></td><td><strong>26.91</strong></td><td><strong>27.20</strong></td><td>-</td><td><strong>26.93</strong></td><td><strong>27.07</strong></td><td><strong>26.82</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>28.47</td><td>28.60</td><td>28.57</td><td>28.38</td><td>28.58</td><td>-</td><td>28.34</td><td>28.47</td><td>28.34</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>27.33</td><td>27.39</td><td>27.28</td><td>27.17</td><td>27.57</td><td>-</td><td>27.32</td><td>27.29</td><td>27.26</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>28.71</td><td>28.80</td><td>28.57</td><td>28.56</td><td>28.75</td><td>-</td><td>28.61</td><td>28.52</td><td>28.48</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>27.85</td><td>28.02</td><td>27.74</td><td>27.69</td><td>27.94</td><td>-</td><td>27.65</td><td>27.77</td><td>27.55</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">pl</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>25.89</td><td>22.01</td><td>26.33</td><td>27.19</td><td>25.99</td><td>26.47</td><td>-</td><td>27.13</td><td>27.36</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>18.26</td><td>18.27</td><td>18.14</td><td>18.21</td><td>17.87</td><td>18.27</td><td>-</td><td>18.29</td><td>18.00</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>17.54</strong></td><td>17.39</td><td><strong>17.32</strong></td><td><strong>17.36</strong></td><td><strong>17.01</strong></td><td>17.43</td><td>-</td><td>17.57</td><td><strong>17.11</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>18.10</td><td>17.96</td><td>17.97</td><td>18.07</td><td>17.47</td><td>18.01</td><td>-</td><td>18.14</td><td>17.67</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>17.56</td><td><strong>17.32</strong></td><td>17.42</td><td>17.45</td><td>17.06</td><td><strong>17.41</strong></td><td>-</td><td><strong>17.50</strong></td><td>17.37</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>18.30</td><td>18.14</td><td>18.02</td><td>18.18</td><td>17.85</td><td>18.17</td><td>-</td><td>18.24</td><td>17.92</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>17.88</td><td>17.55</td><td>17.76</td><td>17.79</td><td>17.57</td><td>17.79</td><td>-</td><td>18.00</td><td>17.70</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">pt</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>19.90</td><td>16.27</td><td>21.74</td><td>20.82</td><td>20.77</td><td>20.99</td><td>20.48</td><td>-</td><td>20.53</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>13.60</td><td>13.59</td><td>13.52</td><td>13.59</td><td>13.38</td><td>13.58</td><td>13.57</td><td>-</td><td>13.34</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>12.37</strong></td><td><strong>12.72</strong></td><td><strong>12.28</strong></td><td><strong>12.37</strong></td><td><strong>12.08</strong></td><td><strong>12.38</strong></td><td><strong>12.40</strong></td><td>-</td><td><strong>12.19</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>13.15</td><td>13.39</td><td>13.10</td><td>13.18</td><td>12.96</td><td>13.16</td><td>13.12</td><td>-</td><td>12.91</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>12.53</td><td><strong>12.72</strong></td><td>12.50</td><td>12.55</td><td>12.30</td><td>12.52</td><td>12.57</td><td>-</td><td>12.30</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>13.12</td><td>13.29</td><td>13.01</td><td>13.12</td><td>12.95</td><td>13.09</td><td>13.06</td><td>-</td><td>12.82</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>12.75</td><td>12.86</td><td>12.63</td><td>12.75</td><td>12.49</td><td>12.77</td><td>12.67</td><td>-</td><td>12.48</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">ro</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>22.32</td><td>15.85</td><td>21.87</td><td>22.04</td><td>23.97</td><td>22.88</td><td>23.63</td><td>22.82</td><td>-</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>14.59</td><td>14.20</td><td>14.52</td><td>14.42</td><td>14.17</td><td>14.59</td><td>14.49</td><td>14.65</td><td>-</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>13.64</strong></td><td><strong>13.29</strong></td><td><strong>13.51</strong></td><td><strong>13.46</strong></td><td><strong>13.38</strong></td><td><strong>13.61</strong></td><td><strong>13.54</strong></td><td><strong>13.72</strong></td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>14.64</td><td>14.34</td><td>14.54</td><td>14.40</td><td>14.36</td><td>14.62</td><td>14.56</td><td>14.67</td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>13.99</td><td>13.63</td><td>13.93</td><td>13.82</td><td>13.62</td><td>13.98</td><td>13.88</td><td>14.01</td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>15.04</td><td>14.96</td><td>14.91</td><td>14.92</td><td>14.77</td><td>15.05</td><td>14.99</td><td>15.11</td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>14.09</td><td>13.74</td><td>13.97</td><td>13.88</td><td>13.73</td><td>14.05</td><td>13.91</td><td>14.13</td><td>-</td>
    </tr>
  </tbody>
</table>


### BLEU

<table>
  <thead>
    <tr>
      <th rowspan="2">SRC\TGT</th>
      <th rowspan="2" align="left">Model</th>
      <th colspan="9">BLEU &#8593;</th>
    </tr>
    <tr>
      <th>de</th><th>en</th><th>es</th><th>fr</th><th>it</th><th>nl</th><th>pl</th><th>pt</th><th>ro</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <th rowspan="7">de</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>-</td><td>17.5</td><td>13.3</td><td>12.1</td><td>8.7</td><td>16.2</td><td>5.9</td><td>12.4</td><td>8.3</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>-</td><td>11.0</td><td>3.7</td><td>3.3</td><td>1.1</td><td>4.1</td><td>1.6</td><td>4.0</td><td>2.2</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td>-</td><td><strong>22.0</strong></td><td><strong>19.7</strong></td><td><strong>20.2</strong></td><td><strong>14.5</strong></td><td><strong>19.0</strong></td><td><strong>8.9</strong></td><td><strong>18.7</strong></td><td><strong>13.5</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→MoE</td>
      <td>-</td><td>12.4</td><td>3.2</td><td>2.1</td><td>1.0</td><td>2.8</td><td>1.4</td><td>3.7</td><td>2.1</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→T-Bias</td>
      <td>-</td><td>19.6</td><td>18.0</td><td>18.0</td><td>12.5</td><td>16.5</td><td>6.9</td><td>17.0</td><td>11.5</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>-</td><td>11.4</td><td>4.1</td><td>3.0</td><td>1.0</td><td>3.4</td><td>1.5</td><td>4.0</td><td>2.2</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>-</td><td>21.4</td><td>19.2</td><td>19.8</td><td>13.9</td><td>18.9</td><td>8.7</td><td>18.5</td><td><strong>13.5</strong></td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">en</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>15.4</td><td>-</td><td>26.0</td><td>24.6</td><td>19.0</td><td>21.9</td><td>9.7</td><td>23.1</td><td>19.8</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>4.0</td><td>-</td><td>9.7</td><td>6.5</td><td>3.1</td><td>5.3</td><td>1.6</td><td>7.1</td><td>4.5</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>20.1</strong></td><td>-</td><td><strong>33.4</strong></td><td>30.7</td><td><strong>25.0</strong></td><td><strong>25.4</strong></td><td><strong>14.7</strong></td><td><strong>29.4</strong></td><td><strong>26.3</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→MoE</td>
      <td>3.8</td><td>-</td><td>9.9</td><td>6.0</td><td>3.1</td><td>4.6</td><td>1.3</td><td>7.5</td><td>4.1</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→T-Bias</td>
      <td>17.4</td><td>-</td><td>30.0</td><td>27.3</td><td>22.4</td><td>22.1</td><td>11.2</td><td>26.2</td><td>21.1</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>3.1</td><td>-</td><td>10.6</td><td>6.1</td><td>2.9</td><td>4.5</td><td>1.4</td><td>7.5</td><td>4.1</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>19.6</td><td>-</td><td>32.5</td><td><strong>30.9</strong></td><td>24.4</td><td>24.5</td><td>14.3</td><td>28.7</td><td>25.8</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">es</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>9.9</td><td>22.1</td><td>-</td><td>20.2</td><td>15.7</td><td>15.1</td><td>6.9</td><td>22.4</td><td>12.2</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>2.1</td><td>13.4</td><td>-</td><td>3.9</td><td>1.5</td><td>3.1</td><td>0.9</td><td>5.4</td><td>2.2</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>13.7</strong></td><td><strong>26.1</strong></td><td>-</td><td>26.3</td><td><strong>21.0</strong></td><td><strong>19.4</strong></td><td><strong>10.3</strong></td><td><strong>26.6</strong></td><td><strong>17.7</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→MoE</td>
      <td>2.0</td><td>13.9</td><td>-</td><td>3.5</td><td>1.6</td><td>2.3</td><td>1.1</td><td>5.0</td><td>1.9</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→T-Bias</td>
      <td>11.8</td><td>23.3</td><td>-</td><td>23.7</td><td>18.4</td><td>17.3</td><td>7.7</td><td>23.7</td><td>14.8</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>1.8</td><td>13.2</td><td>-</td><td>3.7</td><td>1.4</td><td>2.5</td><td>1.1</td><td>5.5</td><td>2.2</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>13.3</td><td>25.1</td><td>-</td><td><strong>26.5</strong></td><td>20.6</td><td>19.3</td><td>10.2</td><td>26.1</td><td>17.5</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">fr</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>11.0</td><td>23.5</td><td>20.3</td><td>-</td><td>17.6</td><td>16.9</td><td>7.4</td><td>23.3</td><td>13.0</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>2.9</td><td>11.9</td><td>6.4</td><td>-</td><td>2.2</td><td>4.0</td><td>1.3</td><td>6.5</td><td>2.4</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>14.9</strong></td><td><strong>28.6</strong></td><td><strong>27.0</strong></td><td>-</td><td><strong>22.5</strong></td><td><strong>21.3</strong></td><td><strong>11.1</strong></td><td><strong>27.5</strong></td><td>18.3</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→MoE</td>
      <td>2.6</td><td>13.8</td><td>5.2</td><td>-</td><td>2.0</td><td>3.0</td><td>1.3</td><td>5.9</td><td>2.1</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→T-Bias</td>
      <td>13.3</td><td>24.8</td><td>24.7</td><td>-</td><td>20.1</td><td>18.5</td><td>9.0</td><td>25.0</td><td>15.2</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>2.3</td><td>11.8</td><td>6.8</td><td>-</td><td>2.2</td><td>3.4</td><td>1.5</td><td>6.1</td><td>2.3</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>14.1</td><td>27.3</td><td>25.9</td><td>-</td><td>22.4</td><td>20.2</td><td>10.9</td><td>26.9</td><td><strong>18.6</strong></td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">it</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>11.3</td><td>23.0</td><td>21.3</td><td>20.3</td><td>-</td><td>16.1</td><td>8.3</td><td>22.4</td><td>13.4</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>2.9</td><td>14.7</td><td>5.1</td><td>4.0</td><td>-</td><td>3.2</td><td>1.7</td><td>5.6</td><td>2.0</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>14.8</strong></td><td><strong>27.0</strong></td><td><strong>27.3</strong></td><td><strong>25.3</strong></td><td>-</td><td><strong>20.2</strong></td><td>11.0</td><td>26.1</td><td><strong>17.8</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→MoE</td>
      <td>2.6</td><td>16.8</td><td>4.3</td><td>3.0</td><td>-</td><td>2.6</td><td>1.6</td><td>5.2</td><td>1.7</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→T-Bias</td>
      <td>12.9</td><td>23.5</td><td>24.9</td><td>22.7</td><td>-</td><td>17.3</td><td>8.8</td><td>24.1</td><td>14.2</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>2.1</td><td>15.2</td><td>5.1</td><td>3.5</td><td>-</td><td>2.8</td><td>1.7</td><td>5.4</td><td>1.8</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>14.0</td><td>26.0</td><td>26.8</td><td>25.1</td><td>-</td><td>19.4</td><td><strong>11.2</strong></td><td><strong>26.5</strong></td><td>17.7</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">nl</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>7.1</td><td>15.6</td><td>11.3</td><td>10.4</td><td>7.3</td><td>-</td><td>3.7</td><td>10.4</td><td>6.3</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>2.3</td><td>9.8</td><td>3.1</td><td>2.6</td><td>1.2</td><td>-</td><td>0.9</td><td>2.9</td><td>1.9</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>12.1</strong></td><td><strong>21.0</strong></td><td><strong>17.6</strong></td><td>16.5</td><td><strong>13.6</strong></td><td>-</td><td><strong>7.0</strong></td><td>16.9</td><td>11.6</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→MoE</td>
      <td>1.7</td><td>12.0</td><td>2.1</td><td>1.8</td><td>0.9</td><td>-</td><td>0.6</td><td>2.1</td><td>1.2</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→T-Bias</td>
      <td>10.1</td><td>18.1</td><td>16.3</td><td>15.3</td><td>11.7</td><td>-</td><td>4.9</td><td>15.5</td><td>10.0</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>1.6</td><td>10.8</td><td>2.8</td><td>2.6</td><td>1.0</td><td>-</td><td>1.1</td><td>2.7</td><td>1.4</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>11.8</td><td>20.0</td><td>17.5</td><td><strong>16.8</strong></td><td>12.8</td><td>-</td><td>6.7</td><td><strong>17.0</strong></td><td><strong>11.8</strong></td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">pl</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>9.5</td><td>19.3</td><td>17.1</td><td>15.7</td><td>11.9</td><td>14.3</td><td>-</td><td>14.6</td><td>10.0</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>2.4</td><td>12.1</td><td>4.6</td><td>3.8</td><td>1.6</td><td>3.4</td><td>-</td><td>3.8</td><td>2.1</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>14.3</strong></td><td><strong>23.9</strong></td><td><strong>24.1</strong></td><td><strong>22.9</strong></td><td><strong>18.6</strong></td><td><strong>19.5</strong></td><td>-</td><td>20.8</td><td><strong>16.5</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→MoE</td>
      <td>2.2</td><td>13.9</td><td>3.7</td><td>3.0</td><td>1.5</td><td>2.2</td><td>-</td><td>3.2</td><td>1.4</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→T-Bias</td>
      <td>12.3</td><td>21.5</td><td>22.1</td><td>21.1</td><td>16.5</td><td>17.6</td><td>-</td><td>19.4</td><td>13.4</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>2.1</td><td>11.6</td><td>4.9</td><td>4.2</td><td>1.3</td><td>2.9</td><td>-</td><td>4.0</td><td>2.0</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>13.5</td><td>23.3</td><td>22.6</td><td>22.3</td><td>18.0</td><td>19.2</td><td>-</td><td><strong>20.9</strong></td><td>16.2</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">pt</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>10.9</td><td>23.7</td><td>22.1</td><td>21.3</td><td>17.3</td><td>15.6</td><td>7.5</td><td>-</td><td>13.9</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>2.3</td><td>13.3</td><td>6.6</td><td>4.0</td><td>1.9</td><td>3.0</td><td>1.1</td><td>-</td><td>2.5</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>15.4</strong></td><td><strong>28.1</strong></td><td><strong>28.3</strong></td><td>27.0</td><td><strong>22.8</strong></td><td><strong>19.7</strong></td><td>10.5</td><td>-</td><td><strong>19.0</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→MoE</td>
      <td>1.9</td><td>17.4</td><td>4.7</td><td>3.4</td><td>1.4</td><td>2.1</td><td>1.0</td><td>-</td><td>1.7</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→T-Bias</td>
      <td>13.6</td><td>24.9</td><td>25.6</td><td>24.9</td><td>20.7</td><td>18.0</td><td>8.8</td><td>-</td><td>16.1</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>1.7</td><td>14.6</td><td>6.7</td><td>3.9</td><td>1.4</td><td>2.3</td><td>1.2</td><td>-</td><td>1.8</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>14.5</td><td>26.7</td><td>27.5</td><td><strong>27.1</strong></td><td>22.5</td><td>19.0</td><td><strong>10.6</strong></td><td>-</td><td><strong>19.0</strong></td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">ro</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>10.9</td><td>25.3</td><td>21.4</td><td>21.4</td><td>15.8</td><td>16.0</td><td>7.9</td><td>18.8</td><td>-</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>1.9</td><td>16.4</td><td>4.6</td><td>3.7</td><td>1.5</td><td>2.2</td><td>0.7</td><td>3.9</td><td>-</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>15.8</strong></td><td><strong>30.1</strong></td><td><strong>28.9</strong></td><td><strong>28.4</strong></td><td><strong>22.1</strong></td><td><strong>19.7</strong></td><td><strong>12.2</strong></td><td><strong>25.3</strong></td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→MoE</td>
      <td>1.9</td><td>17.5</td><td>4.1</td><td>3.4</td><td>1.6</td><td>1.8</td><td>0.7</td><td>3.4</td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE→T-Bias</td>
      <td>13.7</td><td>25.9</td><td>26.3</td><td>25.4</td><td>19.6</td><td>17.6</td><td>9.1</td><td>23.3</td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>1.5</td><td>13.5</td><td>6.5</td><td>3.9</td><td>1.5</td><td>2.3</td><td>0.9</td><td>4.5</td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>15.0</td><td>29.2</td><td>27.7</td><td>28.3</td><td>21.8</td><td>19.2</td><td>11.8</td><td>24.7</td><td>-</td>
    </tr>
  </tbody>
</table>

### COMET

<table>
  <thead>
    <tr>
      <th rowspan="2">SRC\TGT</th>
      <th rowspan="2" align="left">Model</th>
      <th colspan="9">COMET &#8593;</th>
    </tr>
    <tr>
      <th>de</th><th>en</th><th>es</th><th>fr</th><th>it</th><th>nl</th><th>pl</th><th>pt</th><th>ro</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <th rowspan="7">de</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>-</td><td>0.615</td><td>0.531</td><td>0.479</td><td>0.504</td><td>0.549</td><td>0.521</td><td>0.544</td><td>0.545</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>-</td><td>0.522</td><td>0.453</td><td>0.407</td><td>0.409</td><td>0.397</td><td>0.383</td><td>0.447</td><td>0.391</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td>-</td><td><strong>0.683</strong></td><td><strong>0.624</strong></td><td><strong>0.572</strong></td><td><strong>0.591</strong></td><td><strong>0.604</strong></td><td><strong>0.591</strong></td><td><strong>0.636</strong></td><td><strong>0.627</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>-</td><td>0.528</td><td>0.451</td><td>0.400</td><td>0.412</td><td>0.383</td><td>0.377</td><td>0.446</td><td>0.393</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>-</td><td>0.640</td><td>0.590</td><td>0.533</td><td>0.557</td><td>0.546</td><td>0.542</td><td>0.601</td><td>0.572</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>-</td><td>0.524</td><td>0.454</td><td>0.407</td><td>0.408</td><td>0.393</td><td>0.381</td><td>0.448</td><td>0.391</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>-</td><td>0.675</td><td>0.614</td><td>0.565</td><td>0.578</td><td>0.601</td><td>0.578</td><td>0.629</td><td>0.626</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">en</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>0.571</td><td>-</td><td>0.641</td><td>0.606</td><td>0.625</td><td>0.620</td><td>0.584</td><td>0.668</td><td>0.680</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>0.421</td><td>-</td><td>0.533</td><td>0.470</td><td>0.487</td><td>0.430</td><td>0.419</td><td>0.524</td><td>0.458</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>0.638</strong></td><td>-</td><td><strong>0.741</strong></td><td><strong>0.690</strong></td><td><strong>0.714</strong></td><td><strong>0.674</strong></td><td><strong>0.663</strong></td><td><strong>0.749</strong></td><td><strong>0.765</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>0.422</td><td>-</td><td>0.535</td><td>0.467</td><td>0.495</td><td>0.424</td><td>0.421</td><td>0.528</td><td>0.463</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>0.582</td><td>-</td><td>0.696</td><td>0.637</td><td>0.664</td><td>0.612</td><td>0.590</td><td>0.704</td><td>0.693</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>0.413</td><td>-</td><td>0.540</td><td>0.469</td><td>0.488</td><td>0.428</td><td>0.423</td><td>0.532</td><td>0.459</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>0.626</td><td>-</td><td>0.733</td><td><strong>0.690</strong></td><td>0.705</td><td>0.661</td><td>0.649</td><td>0.742</td><td>0.762</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">es</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>0.488</td><td>0.652</td><td>-</td><td>0.548</td><td>0.571</td><td>0.534</td><td>0.546</td><td>0.636</td><td>0.589</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>0.357</td><td>0.536</td><td>-</td><td>0.416</td><td>0.424</td><td>0.385</td><td>0.374</td><td>0.464</td><td>0.396</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>0.544</strong></td><td><strong>0.708</strong></td><td>-</td><td><strong>0.627</strong></td><td><strong>0.657</strong></td><td><strong>0.584</strong></td><td><strong>0.609</strong></td><td><strong>0.709</strong></td><td>0.663</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>0.361</td><td>0.541</td><td>-</td><td>0.416</td><td>0.430</td><td>0.377</td><td>0.375</td><td>0.468</td><td>0.394</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>0.503</td><td>0.669</td><td>-</td><td>0.579</td><td>0.609</td><td>0.539</td><td>0.556</td><td>0.665</td><td>0.608</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>0.358</td><td>0.539</td><td>-</td><td>0.419</td><td>0.425</td><td>0.382</td><td>0.376</td><td>0.466</td><td>0.394</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>0.541</td><td>0.702</td><td>-</td><td>0.622</td><td>0.651</td><td>0.578</td><td>0.600</td><td>0.705</td><td><strong>0.666</strong></td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">fr</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>0.499</td><td>0.685</td><td>0.603</td><td>-</td><td>0.603</td><td>0.551</td><td>0.555</td><td>0.650</td><td>0.618</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>0.373</td><td>0.535</td><td>0.484</td><td>-</td><td>0.440</td><td>0.396</td><td>0.385</td><td>0.481</td><td>0.408</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>0.561</strong></td><td><strong>0.737</strong></td><td><strong>0.700</strong></td><td>-</td><td><strong>0.685</strong></td><td><strong>0.603</strong></td><td><strong>0.616</strong></td><td><strong>0.723</strong></td><td><strong>0.701</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>0.379</td><td>0.552</td><td>0.482</td><td>-</td><td>0.442</td><td>0.391</td><td>0.385</td><td>0.482</td><td>0.410</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>0.519</td><td>0.696</td><td>0.658</td><td>-</td><td>0.636</td><td>0.555</td><td>0.566</td><td>0.674</td><td>0.637</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>0.372</td><td>0.538</td><td>0.485</td><td>-</td><td>0.440</td><td>0.393</td><td>0.385</td><td>0.484</td><td>0.411</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>0.553</td><td>0.730</td><td>0.685</td><td>-</td><td>0.673</td><td>0.592</td><td>0.605</td><td>0.711</td><td>0.695</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">it</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>0.507</td><td>0.679</td><td>0.614</td><td>0.569</td><td>-</td><td>0.551</td><td>0.568</td><td>0.650</td><td>0.623</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>0.372</td><td>0.560</td><td>0.477</td><td>0.425</td><td>-</td><td>0.393</td><td>0.380</td><td>0.472</td><td>0.404</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>0.560</strong></td><td><strong>0.728</strong></td><td><strong>0.698</strong></td><td><strong>0.640</strong></td><td>-</td><td><strong>0.600</strong></td><td><strong>0.619</strong></td><td><strong>0.717</strong></td><td>0.686</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>0.374</td><td>0.578</td><td>0.476</td><td>0.428</td><td>-</td><td>0.391</td><td>0.385</td><td>0.478</td><td>0.412</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>0.520</td><td>0.689</td><td>0.657</td><td>0.593</td><td>-</td><td>0.553</td><td>0.570</td><td>0.672</td><td>0.630</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>0.370</td><td>0.572</td><td>0.483</td><td>0.428</td><td>-</td><td>0.393</td><td>0.384</td><td>0.476</td><td>0.411</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>0.558</td><td>0.722</td><td>0.693</td><td>0.635</td><td>-</td><td>0.591</td><td>0.615</td><td>0.711</td><td><strong>0.689</strong></td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">nl</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>0.444</td><td>0.581</td><td>0.500</td><td>0.460</td><td>0.467</td><td>-</td><td>0.486</td><td>0.509</td><td>0.509</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>0.367</td><td>0.508</td><td>0.435</td><td>0.397</td><td>0.402</td><td>-</td><td>0.365</td><td>0.435</td><td>0.380</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>0.538</strong></td><td><strong>0.660</strong></td><td><strong>0.595</strong></td><td><strong>0.544</strong></td><td><strong>0.561</strong></td><td>-</td><td><strong>0.556</strong></td><td><strong>0.604</strong></td><td><strong>0.593</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>0.359</td><td>0.532</td><td>0.431</td><td>0.395</td><td>0.403</td><td>-</td><td>0.367</td><td>0.432</td><td>0.378</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>0.493</td><td>0.615</td><td>0.556</td><td>0.510</td><td>0.527</td><td>-</td><td>0.512</td><td>0.569</td><td>0.549</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>0.361</td><td>0.517</td><td>0.434</td><td>0.398</td><td>0.400</td><td>-</td><td>0.372</td><td>0.433</td><td>0.377</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>0.528</td><td>0.651</td><td>0.582</td><td>0.538</td><td>0.550</td><td>-</td><td>0.545</td><td>0.601</td><td>0.587</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">pl</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>0.515</td><td>0.643</td><td>0.568</td><td>0.518</td><td>0.545</td><td>0.543</td><td>-</td><td>0.584</td><td>0.583</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>0.385</td><td>0.539</td><td>0.469</td><td>0.424</td><td>0.429</td><td>0.397</td><td>-</td><td>0.462</td><td>0.401</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>0.584</strong></td><td><strong>0.709</strong></td><td><strong>0.667</strong></td><td><strong>0.612</strong></td><td><strong>0.651</strong></td><td><strong>0.608</strong></td><td>-</td><td><strong>0.683</strong></td><td><strong>0.677</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>0.380</td><td>0.552</td><td>0.464</td><td>0.419</td><td>0.431</td><td>0.388</td><td>-</td><td>0.462</td><td>0.402</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>0.539</td><td>0.669</td><td>0.633</td><td>0.572</td><td>0.605</td><td>0.561</td><td>-</td><td>0.645</td><td>0.624</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>0.379</td><td>0.531</td><td>0.466</td><td>0.421</td><td>0.427</td><td>0.395</td><td>-</td><td>0.461</td><td>0.402</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>0.575</td><td>0.698</td><td>0.655</td><td>0.596</td><td>0.634</td><td>0.601</td><td>-</td><td>0.670</td><td>0.669</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">pt</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>0.522</td><td>0.692</td><td>0.631</td><td>0.584</td><td>0.605</td><td>0.556</td><td>0.576</td><td>-</td><td>0.636</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>0.381</td><td>0.557</td><td>0.491</td><td>0.439</td><td>0.444</td><td>0.402</td><td>0.390</td><td>-</td><td>0.409</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>0.581</strong></td><td><strong>0.744</strong></td><td><strong>0.722</strong></td><td><strong>0.662</strong></td><td><strong>0.695</strong></td><td><strong>0.609</strong></td><td><strong>0.632</strong></td><td>-</td><td><strong>0.710</strong></td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>0.381</td><td>0.594</td><td>0.482</td><td>0.436</td><td>0.444</td><td>0.397</td><td>0.392</td><td>-</td><td>0.409</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>0.541</td><td>0.708</td><td>0.684</td><td>0.623</td><td>0.654</td><td>0.566</td><td>0.585</td><td>-</td><td>0.654</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>0.377</td><td>0.572</td><td>0.489</td><td>0.438</td><td>0.440</td><td>0.396</td><td>0.387</td><td>-</td><td>0.409</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>0.577</td><td>0.738</td><td>0.710</td><td>0.656</td><td>0.688</td><td>0.604</td><td>0.624</td><td>-</td><td>0.708</td>
    </tr>
  </tbody>

  <tbody>
    <tr>
      <th rowspan="7">ro</th>
      <td align="left">HENT-SRT-M20&times;9</td>
      <td>0.514</td><td>0.697</td><td>0.606</td><td>0.575</td><td>0.596</td><td>0.563</td><td>0.569</td><td>0.627</td><td>-</td>
    </tr>
    <tr>
      <td align="left">HENT-SRT-M2M</td>
      <td>0.381</td><td>0.585</td><td>0.487</td><td>0.443</td><td>0.446</td><td>0.408</td><td>0.386</td><td>0.480</td><td>-</td>
    </tr>
    <tr>
      <td align="left">LCMA-SRT</td>
      <td><strong>0.587</strong></td><td><strong>0.753</strong></td><td><strong>0.711</strong></td><td><strong>0.667</strong></td><td><strong>0.696</strong></td><td><strong>0.624</strong></td><td><strong>0.642</strong></td><td><strong>0.724</strong></td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td>
      <td>0.384</td><td>0.591</td><td>0.486</td><td>0.443</td><td>0.449</td><td>0.405</td><td>0.388</td><td>0.484</td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td>
      <td>0.537</td><td>0.713</td><td>0.665</td><td>0.616</td><td>0.644</td><td>0.574</td><td>0.574</td><td>0.680</td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td>
      <td>0.377</td><td>0.563</td><td>0.488</td><td>0.441</td><td>0.447</td><td>0.405</td><td>0.386</td><td>0.480</td><td>-</td>
    </tr>
    <tr>
      <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td>
      <td>0.582</td><td>0.747</td><td>0.699</td><td>0.662</td><td>0.687</td><td>0.618</td><td>0.628</td><td>0.714</td><td>-</td>
    </tr>
  </tbody>
</table>

### LMR

<table> <thead> <tr> <th rowspan="2">SRC\TGT</th> <th rowspan="2" align="left">Model</th> <th colspan="9">LMR (%) &#8595;</th> </tr> <tr> <th>de</th><th>en</th><th>es</th><th>fr</th><th>it</th><th>nl</th><th>pl</th><th>pt</th><th>ro</th> </tr> </thead> <tbody> <tr> <th rowspan="7">de</th> <td align="left">HENT-SRT-M20&times;9</td> <td>-</td><td><strong>0.08</strong></td><td>0.70</td><td>0.64</td><td>0.66</td><td>1.00</td><td><strong>0.00</strong></td><td>3.39</td><td>1.70</td> </tr> <tr> <td align="left">HENT-SRT-M2M</td> <td>-</td><td>56.60</td><td>87.83</td><td>90.14</td><td>94.98</td><td>77.09</td><td>83.11</td><td>83.75</td><td>82.55</td> </tr> <tr> <td align="left">LCMA-SRT</td> <td>-</td><td>0.38</td><td><strong>0.49</strong></td><td><strong>0.43</strong></td><td><strong>0.58</strong></td><td><strong>0.84</strong></td><td>0.73</td><td><strong>2.38</strong></td><td>1.79</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td> <td>-</td><td>44.09</td><td>91.70</td><td>93.43</td><td>96.22</td><td>87.51</td><td>88.57</td><td>86.63</td><td>84.42</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td> <td>-</td><td>0.34</td><td>0.56</td><td>0.64</td><td>0.82</td><td>1.84</td><td>0.36</td><td>2.89</td><td><strong>1.62</strong></td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td> <td>-</td><td>54.96</td><td>87.83</td><td>91.79</td><td>97.04</td><td>83.22</td><td>80.86</td><td>83.02</td><td>83.12</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td> <td>-</td><td>0.38</td><td>0.84</td><td>0.86</td><td>1.48</td><td>0.92</td><td>0.58</td><td>3.10</td><td><strong>1.62</strong></td> </tr> </tbody> <tbody> <tr> <th rowspan="7">en</th> <td align="left">HENT-SRT-M20&times;9</td> <td><strong>0.00</strong></td><td>-</td><td>0.79</td><td>0.25</td><td>0.35</td><td>0.65</td><td><strong>0.16</strong></td><td><strong>1.98</strong></td><td>1.64</td> </tr> <tr> <td align="left">HENT-SRT-M2M</td> <td>78.21</td><td>-</td><td>78.93</td><td>86.08</td><td>95.93</td><td>78.54</td><td>85.14</td><td>81.62</td><td>80.55</td> </tr> <tr> <td align="left">LCMA-SRT</td> <td>0.08</td><td>-</td><td>0.95</td><td>0.16</td><td><strong>0.09</strong></td><td><strong>0.24</strong></td><td>0.40</td><td><strong>1.98</strong></td><td>1.64</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td> <td>79.25</td><td>-</td><td>82.40</td><td>85.26</td><td>95.75</td><td>81.54</td><td>87.16</td><td>79.24</td><td>79.45</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td> <td>0.32</td><td>-</td><td>0.71</td><td>0.25</td><td>0.27</td><td>0.49</td><td>0.32</td><td>2.38</td><td><strong>1.28</strong></td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td> <td>83.56</td><td>-</td><td>77.82</td><td>86.24</td><td>95.39</td><td>83.48</td><td>84.49</td><td>78.76</td><td>81.74</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td> <td>0.08</td><td>-</td><td><strong>0.55</strong></td><td><strong>0.08</strong></td><td>0.53</td><td>0.40</td><td>0.32</td><td>2.38</td><td>1.83</td> </tr> </tbody> <tbody> <tr> <th rowspan="7">es</th> <td align="left">HENT-SRT-M20&times;9</td> <td><strong>0.18</strong></td><td><strong>0.22</strong></td><td>-</td><td><strong>0.18</strong></td><td>0.46</td><td>1.01</td><td><strong>0.09</strong></td><td><strong>1.19</strong></td><td><strong>0.88</strong></td> </tr> <tr> <td align="left">HENT-SRT-M2M</td> <td>88.51</td><td>58.54</td><td>-</td><td>92.98</td><td>96.76</td><td>89.58</td><td>90.93</td><td>80.07</td><td>86.81</td> </tr> <tr> <td align="left">LCMA-SRT</td> <td>0.54</td><td>0.61</td><td>-</td><td>0.37</td><td>0.65</td><td>0.92</td><td>0.38</td><td>1.29</td><td>1.54</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td> <td>88.87</td><td>50.66</td><td>-</td><td>93.16</td><td>96.85</td><td>92.59</td><td>92.63</td><td>82.09</td><td>88.68</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td> <td>0.36</td><td>0.61</td><td>-</td><td>0.28</td><td><strong>0.37</strong></td><td>1.28</td><td>0.19</td><td>1.47</td><td>1.10</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td> <td>91.56</td><td>55.95</td><td>-</td><td>92.79</td><td>97.59</td><td>91.58</td><td>87.43</td><td>76.19</td><td>85.71</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td> <td>0.27</td><td>0.50</td><td>-</td><td>0.55</td><td>0.74</td><td><strong>0.82</strong></td><td>0.47</td><td>1.47</td><td>1.43</td> </tr> </tbody> <tbody> <tr> <th rowspan="7">fr</th> <td align="left">HENT-SRT-M20&times;9</td> <td><strong>0.00</strong></td><td><strong>0.11</strong></td><td>0.55</td><td>-</td><td><strong>0.00</strong></td><td><strong>0.26</strong></td><td><strong>0.18</strong></td><td>1.55</td><td><strong>1.16</strong></td> </tr> <tr> <td align="left">HENT-SRT-M2M</td> <td>85.18</td><td>71.51</td><td>87.89</td><td>-</td><td>95.60</td><td>86.66</td><td>89.85</td><td>77.71</td><td>84.93</td> </tr> <tr> <td align="left">LCMA-SRT</td> <td><strong>0.00</strong></td><td>0.33</td><td>0.73</td><td>-</td><td>0.38</td><td>0.87</td><td>0.36</td><td>1.91</td><td><strong>1.16</strong></td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td> <td>88.27</td><td>57.97</td><td>90.88</td><td>-</td><td>96.84</td><td>90.66</td><td>90.82</td><td>81.88</td><td>85.76</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td> <td>0.18</td><td>0.39</td><td><strong>0.36</strong></td><td>-</td><td>0.19</td><td>1.31</td><td>0.27</td><td><strong>0.91</strong></td><td>1.37</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td> <td>88.91</td><td>70.85</td><td>87.41</td><td>-</td><td>96.46</td><td>89.61</td><td>88.84</td><td>78.05</td><td>83.02</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td> <td>0.37</td><td>0.44</td><td>0.46</td><td>-</td><td>0.57</td><td>0.61</td><td>0.27</td><td>1.18</td><td>1.90</td> </tr> </tbody> <tbody> <tr> <th rowspan="7">it</th> <td align="left">HENT-SRT-M20&times;9</td> <td>0.11</td><td><strong>0.00</strong></td><td>0.57</td><td><strong>0.23</strong></td><td>-</td><td><strong>0.36</strong></td><td>0.25</td><td>2.20</td><td><strong>0.54</strong></td> </tr> <tr> <td align="left">HENT-SRT-M2M</td> <td>88.54</td><td>57.23</td><td>92.05</td><td>94.25</td><td>-</td><td>91.61</td><td>92.77</td><td>86.59</td><td>90.38</td> </tr> <tr> <td align="left">LCMA-SRT</td> <td>0.11</td><td>0.42</td><td>0.91</td><td><strong>0.23</strong></td><td>-</td><td>0.84</td><td>0.37</td><td>1.62</td><td>0.95</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td> <td>91.92</td><td>41.43</td><td>91.37</td><td>94.13</td><td>-</td><td>94.36</td><td>93.87</td><td>87.40</td><td>89.70</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td> <td>0.44</td><td>0.12</td><td><strong>0.45</strong></td><td>0.45</td><td>-</td><td>0.72</td><td><strong>0.00</strong></td><td>1.73</td><td>1.49</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td> <td>92.36</td><td>47.80</td><td>91.37</td><td>95.94</td><td>-</td><td>93.04</td><td>91.18</td><td>87.51</td><td>89.30</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td> <td><strong>0.00</strong></td><td>0.12</td><td><strong>0.45</strong></td><td>0.79</td><td>-</td><td>0.72</td><td>0.12</td><td><strong>1.50</strong></td><td>0.81</td> </tr> </tbody> <tbody> <tr> <th rowspan="7">nl</th> <td align="left">HENT-SRT-M20&times;9</td> <td><strong>0.09</strong></td><td><strong>0.34</strong></td><td><strong>0.49</strong></td><td><strong>0.49</strong></td><td>0.90</td><td>-</td><td><strong>0.21</strong></td><td>3.18</td><td><strong>1.60</strong></td> </tr> <tr> <td align="left">HENT-SRT-M2M</td> <td>78.25</td><td>56.62</td><td>88.85</td><td>89.02</td><td>93.71</td><td>-</td><td>86.96</td><td>82.91</td><td>81.98</td> </tr> <tr> <td align="left">LCMA-SRT</td> <td>0.19</td><td>0.86</td><td>1.19</td><td>0.79</td><td><strong>0.79</strong></td><td>-</td><td>0.62</td><td><strong>1.80</strong></td><td>1.94</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td> <td>82.38</td><td>35.99</td><td>94.86</td><td>92.08</td><td>95.38</td><td>-</td><td>89.00</td><td>86.94</td><td>86.99</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td> <td>0.47</td><td>0.52</td><td>1.38</td><td>0.69</td><td><strong>0.79</strong></td><td>-</td><td>0.83</td><td>4.03</td><td>1.71</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td> <td>83.47</td><td>49.05</td><td>88.42</td><td>89.30</td><td>95.72</td><td>-</td><td>81.93</td><td>84.82</td><td>82.65</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td> <td>0.19</td><td>0.63</td><td>0.99</td><td>1.58</td><td>2.02</td><td>-</td><td>1.04</td><td>2.23</td><td>2.17</td> </tr> </tbody> <tbody> <tr> <th rowspan="7">pl</th> <td align="left">HENT-SRT-M20&times;9</td> <td><strong>0.00</strong></td><td><strong>0.22</strong></td><td>0.72</td><td><strong>0.16</strong></td><td><strong>0.42</strong></td><td>1.80</td><td>-</td><td>2.08</td><td>1.72</td> </tr> <tr> <td align="left">HENT-SRT-M2M</td> <td>82.54</td><td>60.55</td><td>89.70</td><td>91.34</td><td>96.01</td><td>86.68</td><td>-</td><td>83.45</td><td>87.29</td> </tr> <tr> <td align="left">LCMA-SRT</td> <td><strong>0.00</strong></td><td>0.40</td><td>0.80</td><td>0.48</td><td>0.85</td><td><strong>1.14</strong></td><td>-</td><td>1.60</td><td><strong>1.01</strong></td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td> <td>85.89</td><td>46.27</td><td>92.66</td><td>92.05</td><td>96.94</td><td>90.28</td><td>-</td><td>88.57</td><td>88.70</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td> <td>0.39</td><td>0.49</td><td>1.04</td><td>0.32</td><td>0.51</td><td>1.47</td><td>-</td><td><strong>1.44</strong></td><td>1.31</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td> <td>86.52</td><td>64.23</td><td>89.31</td><td>91.49</td><td>97.28</td><td>89.13</td><td>-</td><td>82.57</td><td>86.28</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td> <td><strong>0.00</strong></td><td>0.36</td><td><strong>0.40</strong></td><td>0.79</td><td>0.85</td><td>1.39</td><td>-</td><td>1.60</td><td>1.21</td> </tr> </tbody> <tbody> <tr> <th rowspan="7">pt</th> <td align="left">HENT-SRT-M20&times;9</td> <td><strong>0.00</strong></td><td><strong>0.17</strong></td><td>0.16</td><td><strong>0.16</strong></td><td>0.08</td><td><strong>0.41</strong></td><td><strong>0.08</strong></td><td>-</td><td><strong>0.81</strong></td> </tr> <tr> <td align="left">HENT-SRT-M2M</td> <td>88.36</td><td>65.79</td><td>86.39</td><td>91.99</td><td>97.51</td><td>90.31</td><td>90.80</td><td>-</td><td>87.09</td> </tr> <tr> <td align="left">LCMA-SRT</td> <td>0.16</td><td>0.39</td><td><strong>0.08</strong></td><td>0.24</td><td>0.08</td><td>0.81</td><td>0.50</td><td>-</td><td>1.17</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td> <td>90.32</td><td>40.07</td><td>92.60</td><td>94.50</td><td>97.68</td><td>94.79</td><td>92.56</td><td>-</td><td>91.43</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td> <td>0.24</td><td>0.48</td><td>0.24</td><td>0.24</td><td><strong>0.00</strong></td><td>0.90</td><td><strong>0.08</strong></td><td>-</td><td><strong>0.81</strong></td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td> <td>91.50</td><td>54.86</td><td>85.19</td><td>92.93</td><td>98.76</td><td>93.65</td><td>90.22</td><td>-</td><td>89.89</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td> <td><strong>0.00</strong></td><td>0.26</td><td><strong>0.08</strong></td><td>0.47</td><td>0.17</td><td>0.49</td><td>0.25</td><td>-</td><td>1.08</td> </tr> </tbody> <tbody> <tr> <th rowspan="7">ro</th> <td align="left">HENT-SRT-M20&times;9</td> <td><strong>0.00</strong></td><td><strong>0.15</strong></td><td><strong>0.42</strong></td><td>0.26</td><td>0.77</td><td>0.91</td><td><strong>0.00</strong></td><td>1.58</td><td>-</td> </tr> <tr> <td align="left">HENT-SRT-M2M</td> <td>90.98</td><td>50.74</td><td>93.27</td><td>95.33</td><td>98.46</td><td>93.14</td><td>93.04</td><td>89.42</td><td>-</td> </tr> <tr> <td align="left">LCMA-SRT</td> <td>0.16</td><td>0.56</td><td>0.50</td><td><strong>0.17</strong></td><td>0.77</td><td>1.24</td><td>0.43</td><td><strong>1.33</strong></td><td>-</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;MoE</td> <td>91.39</td><td>45.21</td><td>93.69</td><td>94.99</td><td>97.52</td><td>94.38</td><td>94.93</td><td>89.92</td><td>-</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;TGT-MoE&#8594;T-Bias</td> <td>0.08</td><td>0.31</td><td><strong>0.42</strong></td><td><strong>0.17</strong></td><td><strong>0.34</strong></td><td>0.83</td><td>0.17</td><td>2.00</td><td>-</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o TGT-MoE</td> <td>92.61</td><td>65.55</td><td>87.96</td><td>94.64</td><td>98.80</td><td>92.81</td><td>91.92</td><td>86.75</td><td>-</td> </tr> <tr> <td align="left">&nbsp;&nbsp;&nbsp;w/o SRC-MoE</td> <td>0.16</td><td>0.31</td><td>0.58</td><td>0.26</td><td>1.03</td><td><strong>0.41</strong></td><td>0.60</td><td>2.00</td><td>-</td> </tr> </tbody> </table>
